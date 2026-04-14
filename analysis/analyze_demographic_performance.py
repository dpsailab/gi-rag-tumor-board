"""
analyze_demographic_performance.py
====================================
Exploratory analysis of concordance performance by demographic variables
(age and sex) across all 16 RAG framework configurations.

This analysis addresses the equity dimension raised in peer review:
whether AI performance differences correlate with patient demographic
characteristics, which could exacerbate existing healthcare disparities
if deployed uncritically.

Analyses performed:
    1. Age distribution by concordance group (high vs low performers)
    2. Sex distribution by concordance group
    3. Mean concordance by sex across all configurations
    4. Age correlation with mean concordance (Spearman)
    5. Subgroup analysis: age + sex × optimal configuration concordance
    6. Exploratory: age × tumour type interaction on concordance

Statistical approach:
    - Mann-Whitney U test for age differences between groups (non-parametric,
      appropriate for non-normally distributed age data with n=100)
    - Chi-square test for sex distribution differences
    - Spearman correlation for age vs concordance (ordinal-safe)
    - All tests two-tailed, α = 0.05
    - Given small subgroup sizes (n=20 per tumour type), results are
      interpreted as exploratory and hypothesis-generating only

Note: with n=100 total cases, statistical power for detecting small effects
of demographic variables is limited. Results should be interpreted cautiously.

Outputs:
    - tables/demographic_performance_summary.csv
    - tables/demographic_concordance_by_sex.csv
    - tables/demographic_age_correlation.csv
    - img/demographic_performance_figure.png
    - Console summary for manuscript
"""

import ast
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

# =========================
# Configuration
# =========================

DATA_PATH = "../data/anonymized_dataset/Tumorboard_ChatGPT_anomyzed_dataset.csv"
TABLE_DIR = "tables"
IMG_DIR   = "img"

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

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

# Optimal configuration (best performing)
OPTIMAL_COL = 'rw-RAG w/ Selected Corpora_4o'

# Age grouping boundaries (tertiles will be computed from data)
AGE_COL    = "age"
SEX_COL    = "gender"   # values: 'm' / 'w'
TUMOUR_COL = "tumour_type"


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
# Main analysis
# =========================

if __name__ == "__main__":

    df = load_data(DATA_PATH)

    # ------------------------------------------------------------------
    # 1. Compute mean concordance across all 16 configurations per case
    # ------------------------------------------------------------------
    concordance_cols = [f"{col}_treatment_concordance" for col in MODEL_COLS]
    df["mean_concordance"] = df[concordance_cols].mean(axis=1)
    df["optimal_concordance"] = df[f"{OPTIMAL_COL}_treatment_concordance"]

    # Normalize sex column
    df[SEX_COL] = df[SEX_COL].astype(str).str.strip().str.lower()
    df["sex_label"] = df[SEX_COL].map({"m": "Male", "w": "Female"})

    # Age tertile groups
    df["age_tertile"] = pd.qcut(df[AGE_COL], q=3, labels=["Younger", "Middle", "Older"])

    print(f"\nAge: median={df[AGE_COL].median():.1f}, "
          f"IQR={df[AGE_COL].quantile(0.25):.1f}–{df[AGE_COL].quantile(0.75):.1f}")
    print(f"Sex: {df['sex_label'].value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # 2. Mean concordance by sex
    # ------------------------------------------------------------------
    print("\n=== Mean Concordance by Sex (across all 16 configurations) ===")
    sex_concordance = df.groupby("sex_label")["mean_concordance"].agg(["mean", "std", "count"])
    sex_concordance.columns = ["Mean", "SD", "N"]
    sex_concordance["Mean_%"] = (sex_concordance["Mean"] * 100).round(1)
    sex_concordance["SD_%"]   = (sex_concordance["SD"]   * 100).round(1)
    print(sex_concordance[["N", "Mean_%", "SD_%"]].to_string())

    # Mann-Whitney U test: sex vs mean concordance
    male_conc   = df[df["sex_label"] == "Male"]["mean_concordance"].dropna()
    female_conc = df[df["sex_label"] == "Female"]["mean_concordance"].dropna()

    if len(male_conc) > 0 and len(female_conc) > 0:
        u_stat, p_mw = stats.mannwhitneyu(male_conc, female_conc, alternative="two-sided")
        print(f"\nMann-Whitney U (sex vs mean concordance): U={u_stat:.1f}, p={p_mw:.3f}")
    else:
        p_mw = None
        print("\nInsufficient data for Mann-Whitney U test.")

    sex_concordance.to_csv(os.path.join(TABLE_DIR, "demographic_concordance_by_sex.csv"))

    # ------------------------------------------------------------------
    # 3. Spearman correlation: age vs mean concordance
    # ------------------------------------------------------------------
    print("\n=== Spearman Correlation: Age vs Mean Concordance ===")
    valid = df[[AGE_COL, "mean_concordance"]].dropna()
    rho, p_spearman = stats.spearmanr(valid[AGE_COL], valid["mean_concordance"])
    print(f"  Spearman rho = {rho:.3f}, p = {p_spearman:.3f} (n={len(valid)})")

    # Also for optimal configuration only
    valid_opt = df[[AGE_COL, "optimal_concordance"]].dropna()
    rho_opt, p_opt = stats.spearmanr(valid_opt[AGE_COL], valid_opt["optimal_concordance"])
    print(f"  Optimal config only: rho = {rho_opt:.3f}, p = {p_opt:.3f}")

    age_corr_df = pd.DataFrame({
        "Comparison": ["Age vs mean concordance (all configs)",
                       "Age vs concordance (optimal config only)"],
        "Spearman_rho": [round(rho, 3), round(rho_opt, 3)],
        "p_value": [round(p_spearman, 3), round(p_opt, 3)],
        "N": [len(valid), len(valid_opt)]
    })
    age_corr_df.to_csv(os.path.join(TABLE_DIR, "demographic_age_correlation.csv"), index=False)

    # ------------------------------------------------------------------
    # 4. Mean concordance by sex AND tumour type
    # ------------------------------------------------------------------
    print("\n=== Mean Concordance by Sex × Tumour Type (optimal config) ===")
    sex_tumour = df.groupby([TUMOUR_COL, "sex_label"])["optimal_concordance"].agg(
        ["mean", "count"]
    ).round(3)
    sex_tumour["mean_%"] = (sex_tumour["mean"] * 100).round(1)
    print(sex_tumour[["count", "mean_%"]].to_string())

    # ------------------------------------------------------------------
    # 5. Mean concordance by age tertile
    # ------------------------------------------------------------------
    print("\n=== Mean Concordance by Age Tertile ===")
    age_tertile = df.groupby("age_tertile")["mean_concordance"].agg(["mean", "std", "count"])
    age_tertile["Mean_%"] = (age_tertile["mean"] * 100).round(1)
    age_tertile["SD_%"]   = (age_tertile["std"]  * 100).round(1)
    age_boundaries = df.groupby("age_tertile")[AGE_COL].agg(["min", "max"])
    age_tertile = age_tertile.join(age_boundaries)
    print(age_tertile[["count", "min", "max", "Mean_%", "SD_%"]].to_string())

    # Kruskal-Wallis across age tertiles
    groups = [df[df["age_tertile"] == t]["mean_concordance"].dropna().values
              for t in ["Younger", "Middle", "Older"]]
    if all(len(g) > 0 for g in groups):
        h_stat, p_kruskal = stats.kruskal(*groups)
        print(f"\nKruskal-Wallis (age tertile vs concordance): H={h_stat:.2f}, p={p_kruskal:.3f}")
    else:
        p_kruskal = None

    # ------------------------------------------------------------------
    # 5b. Age and sex distribution by tumour type
    # ------------------------------------------------------------------
    print("\n=== Age and Sex Distribution by Tumour Type ===")
    demo_by_tumour = df.groupby(TUMOUR_COL).agg(
        N=(AGE_COL, "count"),
        Age_median=(AGE_COL, "median"),
        Age_IQR_low=(AGE_COL, lambda x: x.quantile(0.25)),
        Age_IQR_high=(AGE_COL, lambda x: x.quantile(0.75)),
        Male_n=("sex_label", lambda x: (x == "Male").sum()),
        Female_n=("sex_label", lambda x: (x == "Female").sum()),
        Mean_concordance_pct=("mean_concordance", lambda x: round(x.mean() * 100, 1)),
        Optimal_concordance_pct=("optimal_concordance", lambda x: round(x.mean() * 100, 1)),
    ).round(1)
    demo_by_tumour["Male_%"] = (demo_by_tumour["Male_n"] / demo_by_tumour["N"] * 100).round(1)
    print(demo_by_tumour.to_string())
    demo_by_tumour.to_csv(os.path.join(TABLE_DIR, "demographic_by_tumour_type.csv"))

    # Kruskal-Wallis: age distribution differs by tumour type?
    tumour_age_groups = [df[df[TUMOUR_COL] == t][AGE_COL].dropna().values
                         for t in df[TUMOUR_COL].unique()]
    if all(len(g) > 0 for g in tumour_age_groups):
        h_age_tumour, p_age_tumour = stats.kruskal(*tumour_age_groups)
        print(f"\nKruskal-Wallis age across tumour types: H={h_age_tumour:.2f}, p={p_age_tumour:.3f}")

    # Chi-square: sex distribution differs by tumour type?
    sex_tumour_table = pd.crosstab(df[TUMOUR_COL], df["sex_label"])
    chi2_sex, p_chi2_sex, _, _ = stats.chi2_contingency(sex_tumour_table)
    print(f"Chi-square sex × tumour type: χ²={chi2_sex:.2f}, p={p_chi2_sex:.3f}")
    print(sex_tumour_table.to_string())

    # ------------------------------------------------------------------
    # 6. Summary table
    # ------------------------------------------------------------------
    summary_rows = []
    for sex in ["Male", "Female"]:
        sub = df[df["sex_label"] == sex]
        summary_rows.append({
            "Group": f"Sex: {sex}",
            "N": len(sub),
            "Mean_concordance_all_%": round(sub["mean_concordance"].mean() * 100, 1),
            "SD_%": round(sub["mean_concordance"].std() * 100, 1),
            "Optimal_concordance_%": round(sub["optimal_concordance"].mean() * 100, 1),
        })
    for t in ["Younger", "Middle", "Older"]:
        sub = df[df["age_tertile"] == t]
        age_range = f"{sub[AGE_COL].min():.0f}–{sub[AGE_COL].max():.0f}"
        summary_rows.append({
            "Group": f"Age tertile: {t} ({age_range} yrs)",
            "N": len(sub),
            "Mean_concordance_all_%": round(sub["mean_concordance"].mean() * 100, 1),
            "SD_%": round(sub["mean_concordance"].std() * 100, 1),
            "Optimal_concordance_%": round(sub["optimal_concordance"].mean() * 100, 1),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(TABLE_DIR, "demographic_performance_summary.csv"), index=False)
    print(f"\nSummary table saved to {TABLE_DIR}/demographic_performance_summary.csv")

    # ------------------------------------------------------------------
    # 7. Figure — 3 panels
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    COLORS = {"Male": "#4C9BE8", "Female": "#E8834C"}

    # Panel A: Mean concordance by sex (boxplot)
    sex_data = [df[df["sex_label"] == s]["mean_concordance"].dropna().values * 100
                for s in ["Male", "Female"]]
    bp = ax1.boxplot(sex_data, patch_artist=True, widths=0.5,
                     medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], [COLORS["Male"], COLORS["Female"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    np.random.seed(42)
    for i, (data, sex) in enumerate(zip(sex_data, ["Male", "Female"]), start=1):
        jitter = np.random.uniform(-0.12, 0.12, size=len(data))
        ax1.scatter(i + jitter, data, color="black", alpha=0.25, s=10, zorder=3)
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(["Male", "Female"], fontsize=10)
    ax1.set_ylabel("Mean Concordance across\nall configurations (%)", fontsize=9)
    ax1.set_title("A — Concordance by Sex", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 110)
    p_text = f"p={p_mw:.3f}" if p_mw is not None else "p=n/a"
    ax1.text(0.5, 0.97, f"Mann-Whitney U: {p_text}",
             transform=ax1.transAxes, ha="center", va="top", fontsize=8, color="gray")

    # Panel B: Age vs mean concordance (scatter + regression)
    ax2.scatter(df[AGE_COL], df["mean_concordance"] * 100,
                c=df["sex_label"].map(COLORS), alpha=0.6, s=25, zorder=3)
    # Regression line
    x_age = df[AGE_COL].dropna()
    y_conc = df.loc[x_age.index, "mean_concordance"] * 100
    m, b = np.polyfit(x_age, y_conc, 1)
    x_line = np.linspace(x_age.min(), x_age.max(), 100)
    ax2.plot(x_line, m * x_line + b, color="gray", linewidth=1.5, linestyle="--", alpha=0.7)
    ax2.set_xlabel("Age (years)", fontsize=9)
    ax2.set_ylabel("Mean Concordance (%)", fontsize=9)
    ax2.set_title("B — Age vs Concordance", fontsize=10, fontweight="bold")
    ax2.text(0.05, 0.97, f"Spearman ρ={rho:.2f}, p={p_spearman:.3f}",
             transform=ax2.transAxes, ha="left", va="top", fontsize=8, color="gray")
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=COLORS["Male"], label="Male"),
                       Patch(facecolor=COLORS["Female"], label="Female")]
    ax2.legend(handles=legend_elements, fontsize=8, loc="lower right")

    # Panel C: Mean concordance by age tertile (bar)
    tertile_means = [df[df["age_tertile"] == t]["mean_concordance"].mean() * 100
                     for t in ["Younger", "Middle", "Older"]]
    tertile_stds  = [df[df["age_tertile"] == t]["mean_concordance"].std() * 100
                     for t in ["Younger", "Middle", "Older"]]
    tertile_labels = [f"Younger\n(n={len(df[df['age_tertile']=='Younger'])})",
                      f"Middle\n(n={len(df[df['age_tertile']=='Middle'])})",
                      f"Older\n(n={len(df[df['age_tertile']=='Older'])})"]
    ax3.bar([1, 2, 3], tertile_means, yerr=tertile_stds,
            color=["#6DBE6D", "#4C9BE8", "#E8834C"], alpha=0.75,
            capsize=5, width=0.5, error_kw=dict(linewidth=1.2))
    ax3.set_xticks([1, 2, 3])
    ax3.set_xticklabels(tertile_labels, fontsize=8)
    ax3.set_ylabel("Mean Concordance (%)", fontsize=9)
    ax3.set_title("C — Concordance by Age Tertile", fontsize=10, fontweight="bold")
    ax3.set_ylim(0, 110)
    p_kruskal_text = f"p={p_kruskal:.3f}" if p_kruskal is not None else "p=n/a"
    ax3.text(0.5, 0.97, f"Kruskal-Wallis: {p_kruskal_text}",
             transform=ax3.transAxes, ha="center", va="top", fontsize=8, color="gray")

    fig.suptitle("Demographic Correlates of Framework Concordance Performance",
                 fontsize=11, fontweight="bold", y=1.02)

    out_path = os.path.join(IMG_DIR, "demographic_performance_figure.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved to {out_path}")

    # ------------------------------------------------------------------
    # 8. Manuscript-ready summary
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("MANUSCRIPT-READY SUMMARY")
    print("="*70)
    print(f"  Male: n={int(sex_concordance.loc['Male','N'])}, "
          f"mean concordance {sex_concordance.loc['Male','Mean_%']}% "
          f"(SD {sex_concordance.loc['Male','SD_%']}%)")
    print(f"  Female: n={int(sex_concordance.loc['Female','N'])}, "
          f"mean concordance {sex_concordance.loc['Female','Mean_%']}% "
          f"(SD {sex_concordance.loc['Female','SD_%']}%)")
    print(f"  Mann-Whitney U sex comparison: p={p_mw:.3f}" if p_mw else "  Sex comparison: n/a")
    print(f"  Spearman age-concordance: rho={rho:.3f}, p={p_spearman:.3f}")
    print(f"  Kruskal-Wallis age tertile: p={p_kruskal:.3f}" if p_kruskal else "  Age tertile: n/a")
