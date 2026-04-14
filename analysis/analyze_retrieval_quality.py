"""
analyze_retrieval_quality.py
=============================
Retrieval quality analysis across RAG configurations, quantifying:
  1. Actionability of retrieved chunks — proportion of retrieved guideline chunks
     classified as clinically actionable (keep=1) versus non-actionable (keep=0)
     across full and selected corpus configurations.
  2. Retrieval similarity scores — mean and minimum cosine similarity of the top-5
     retrieved chunks per case, as a proxy for retrieval confidence.

Background:
    The curated (selected) corpus excludes non-actionable guideline content
    (epidemiology, prevention, screening, reference lists) prior to retrieval.
    By design, selected corpus configurations retrieve only actionable chunks
    (non_actionable_n = 0 by construction). This analysis quantifies the
    complementary question: how much non-actionable content does the full corpus
    retrieve, and how does query formulation (original vs. rewritten vs. translated)
    modulate this noise?

    Retrieval similarity scores provide an additional quality dimension:
    higher mean scores indicate that retrieved chunks are semantically closer
    to the query, regardless of actionability classification.

Configurations analyzed:
    - RAG Full Corpora          (original query, full corpus)
    - RAG Selected Corpora      (original query, curated corpus)
    - rw_RAG Full Corpora       (rewritten query, full corpus)
    - rw_RAG Selected Corpora   (rewritten query, curated corpus)
    - tr_RAG Full Corpora       (translated query, full corpus)
    - tr_RAG Selected Corpora   (translated query, curated corpus)

Expected findings:
    - Selected corpus configurations: non_actionable_n = 0 (by construction)
    - Full corpus configurations: non_actionable_n > 0, modulated by query type
    - Rewritten/translated queries on full corpus: lower non_actionable_n and
      higher similarity scores than original queries, consistent with noise
      reduction through query reformulation

Outputs:
    - Printed summary tables (console)
    - tables/retrieval_quality_summary.csv
    - tables/retrieval_quality_by_tumour_type.csv
    - img/retrieval_quality_figure.png
"""

import ast
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# =========================
# Configuration
# =========================

DATA_PATH = "../data/anonymized_dataset/Tumorboard_ChatGPT_anomyzed_dataset.csv"
TABLE_DIR = "tables"
IMG_DIR   = "img"

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# RAG configurations to analyze (using renamed column names)
RAG_CONFIGS = [
    "RAG Full Corpora",
    "RAG Selected Corpora",
    "rw_RAG Full Corpora",
    "rw_RAG Selected Corpora",
    "tr_RAG Full Corpora",
    "tr_RAG Selected Corpora",
]

# Column name pattern: {prefix}_{config}
# e.g. "Actionable N_RAG Full Corpora"
PREFIX_ACTIONABLE     = "Actionable N_"
PREFIX_NON_ACTIONABLE = "Non-Actionable N_"
PREFIX_SCORE_MEAN     = "Score Mean_"
PREFIX_SCORE_MIN      = "Score Min_"

# Plot colors per configuration
CONFIG_COLORS = {
    "RAG Full Corpora":         "#A0A0A0",   # grey
    "RAG Selected Corpora":     "#4C9BE8",   # blue
    "rw_RAG Full Corpora":      "#E8834C",   # orange
    "rw_RAG Selected Corpora":  "#D94F4F",   # red
    "tr_RAG Full Corpora":      "#6DBE6D",   # green
    "tr_RAG Selected Corpora":  "#2E8B57",   # dark green
}

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
# Helper: build column name
# =========================

def col(config: str, prefix: str) -> str:
    """Build full column name from prefix and configuration label."""
    return f"{prefix}{config}"


# =========================
# 1. Overall retrieval quality summary
# =========================

def compute_overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean actionable_n, non_actionable_n, score_mean, score_min
    across all 100 cases for each RAG configuration.

    Note: selected corpus configurations will show non_actionable_n = 0
    by construction, since non-actionable chunks are excluded prior to retrieval.
    """
    rows = []
    for config in RAG_CONFIGS:
        c_act = col(config, PREFIX_ACTIONABLE)
        c_nact = col(config, PREFIX_NON_ACTIONABLE)
        c_smean = col(config, PREFIX_SCORE_MEAN)
        c_smin = col(config, PREFIX_SCORE_MIN)

        # Check columns exist
        missing = [c for c in [c_act, c_nact, c_smean, c_smin] if c not in df.columns]
        if missing:
            print(f"  WARNING: missing columns for {config}: {missing}")
            continue

        rows.append({
            "Configuration":       config,
            "Mean Actionable N":   round(df[c_act].mean(),  2),
            "Mean Non-Actionable N": round(df[c_nact].mean(), 2),
            "% Non-Actionable":    round(df[c_nact].mean() / 5 * 100, 1),
            "Mean Score Mean":     round(df[c_smean].mean(), 3),
            "Mean Score Min":      round(df[c_smin].mean(),  3),
            "SD Score Mean":       round(df[c_smean].std(),  3),
        })

    summary = pd.DataFrame(rows)
    print("\n=== Overall Retrieval Quality Summary ===")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(TABLE_DIR, "retrieval_quality_summary.csv"), index=False)
    print(f"\nSaved to {TABLE_DIR}/retrieval_quality_summary.csv")
    return summary


# =========================
# 2. Retrieval quality by tumour type
# =========================

def compute_by_tumour_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean non_actionable_n and mean score_mean per tumour type
    for full corpus configurations only (selected corpus = 0 by construction).
    """
    full_corpus_configs = [c for c in RAG_CONFIGS if "Selected" not in c]

    rows = []
    for tumour in df["tumour_type"].unique():
        sub = df[df["tumour_type"] == tumour]
        for config in full_corpus_configs:
            c_nact = col(config, PREFIX_NON_ACTIONABLE)
            c_smean = col(config, PREFIX_SCORE_MEAN)
            if c_nact not in df.columns:
                continue
            rows.append({
                "Tumour Type":         tumour,
                "Configuration":       config,
                "Mean Non-Actionable N": round(sub[c_nact].mean(), 2),
                "% Non-Actionable":    round(sub[c_nact].mean() / 5 * 100, 1),
                "Mean Score Mean":     round(sub[c_smean].mean(), 3),
                "N Cases":             len(sub),
            })

    by_type = pd.DataFrame(rows)
    print("\n=== Retrieval Quality by Tumour Type (Full Corpus only) ===")
    print(by_type.to_string(index=False))
    by_type.to_csv(os.path.join(TABLE_DIR, "retrieval_quality_by_tumour_type.csv"), index=False)
    print(f"Saved to {TABLE_DIR}/retrieval_quality_by_tumour_type.csv")
    return by_type


# =========================
# 3. Figure
# =========================

def plot_retrieval_quality(df: pd.DataFrame, summary: pd.DataFrame):
    """
    Three-panel figure:
      A — Mean non-actionable chunks per case by configuration (bar chart)
          Selected corpus configs show 0 by construction.
      B — Mean cosine similarity score (mean of top-5) by configuration (bar + scatter)
      C — Distribution of non-actionable chunks for full corpus configs (boxplot)
    """
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    configs      = summary["Configuration"].tolist()
    colors       = [CONFIG_COLORS.get(c, "#888888") for c in configs]
    x_pos        = list(range(len(configs)))
    short_labels = [c.replace("_RAG", "\nRAG").replace(" Corpora", "") for c in configs]

    # --- Panel A: Mean non-actionable N ---
    bars_a = ax1.bar(x_pos, summary["Mean Non-Actionable N"],
                     color=colors, alpha=0.8, width=0.6)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(short_labels, fontsize=7, rotation=30, ha="right")
    ax1.set_ylabel("Mean Non-Actionable Chunks (out of 5)", fontsize=9)
    ax1.set_title("A — Retrieval Noise\n(non-actionable chunks per case)",
                  fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 5)
    ax1.axhline(0, color="gray", linewidth=0.5)
    for bar, val in zip(bars_a, summary["Mean Non-Actionable N"]):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                 f"{val:.2f}", ha="center", fontsize=7)

    # --- Panel B: Mean similarity score ---
    bars_b = ax2.bar(x_pos, summary["Mean Score Mean"],
                     yerr=summary["SD Score Mean"],
                     color=colors, alpha=0.8, width=0.6,
                     capsize=4, error_kw=dict(linewidth=1.2))
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(short_labels, fontsize=7, rotation=30, ha="right")
    ax2.set_ylabel("Mean Cosine Similarity (top-5 chunks)", fontsize=9)
    ax2.set_title("B — Retrieval Confidence\n(mean similarity score ± SD)",
                  fontsize=9, fontweight="bold")
    ax2.set_ylim(0, 1.0)

    # --- Panel C: Per-case non-actionable distribution (full corpus only) ---
    full_configs = [c for c in RAG_CONFIGS if "Selected" not in c]
    box_data     = []
    box_labels   = []
    box_colors   = []
    for config in full_configs:
        c_nact = col(config, PREFIX_NON_ACTIONABLE)
        if c_nact in df.columns:
            box_data.append(df[c_nact].dropna().values)
            box_labels.append(config.replace("_RAG", "\nRAG").replace(" Corpora", ""))
            box_colors.append(CONFIG_COLORS.get(config, "#888888"))

    bp = ax3.boxplot(box_data, patch_artist=True, widths=0.5,
                     medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Jittered individual points
    np.random.seed(42)
    for i, data in enumerate(box_data, start=1):
        jitter = np.random.uniform(-0.15, 0.15, size=len(data))
        ax3.scatter(i + jitter, data, color="black", alpha=0.2, s=8, zorder=3)

    ax3.set_xticks(range(1, len(box_labels) + 1))
    ax3.set_xticklabels(box_labels, fontsize=7, rotation=30, ha="right")
    ax3.set_ylabel("Non-Actionable Chunks per Case (out of 5)", fontsize=9)
    ax3.set_title("C — Per-Case Distribution\n(full corpus configurations only)",
                  fontsize=9, fontweight="bold")
    ax3.set_ylim(-0.2, 5.2)

    fig.suptitle(
        "Retrieval Quality Analysis: Actionability and Similarity Across RAG Configurations",
        fontsize=11, fontweight="bold", y=1.02
    )

    out_path = os.path.join(IMG_DIR, "retrieval_quality_figure.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\nFigure saved to {out_path}")


# =========================
# Main
# =========================

if __name__ == "__main__":

    df = load_data(DATA_PATH)

    input(df.columns)

    # Convert metric columns to numeric (stored as strings in some Excel exports)
    all_metric_cols = []
    for config in RAG_CONFIGS:
        for prefix in [PREFIX_ACTIONABLE, PREFIX_NON_ACTIONABLE,
                       PREFIX_SCORE_MEAN, PREFIX_SCORE_MIN]:
            c = col(config, prefix)
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                all_metric_cols.append(c)

    print(f"\nMetric columns found: {len(all_metric_cols)} / {len(RAG_CONFIGS) * 4} expected")

    # 1. Overall summary
    summary = compute_overall_summary(df)

    # 2. By tumour type
    by_type = compute_by_tumour_type(df)

    # 3. Figure
    plot_retrieval_quality(df, summary)

    print("\nDone.")
