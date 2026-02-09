"""
Analysis of demographics  – Reproducible Analysis Script
"""

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load anonymized dataset."""
    df = pd.read_csv(path)
    return df


def age_statistics(df: pd.DataFrame, group_col: str = None) -> pd.DataFrame:
    """Compute median age and IQR overall or by subgroup."""
    if group_col:
        stats = df.groupby(group_col)["age"].agg(
            median="median",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75)
        )
    else:
        stats = pd.DataFrame({
            "median": [df["age"].median()],
            "q1": [df["age"].quantile(0.25)],
            "q3": [df["age"].quantile(0.75)]
        }, index=["Overall"])

    stats["IQR"] = stats["q3"] - stats["q1"]
    stats["median_IQR"] = (
        stats["median"].round(1).astype(str)
        + " ("
        + stats["q1"].round(0).astype(int).astype(str)
        + "–"
        + stats["q3"].round(0).astype(int).astype(str)
        + ")"
    )
    return stats[["median_IQR"]]


def gender_distribution(df: pd.DataFrame, group_col: str = None) -> pd.DataFrame:
    """Compute gender counts and percentages."""
    if group_col:
        counts = df.groupby(group_col)["gender"].value_counts()
        perc = df.groupby(group_col)["gender"].value_counts(normalize=True) * 100
    else:
        counts = df["gender"].value_counts()
        perc = df["gender"].value_counts(normalize=True) * 100

    out = pd.concat([counts, perc], axis=1)
    out.columns = ["N", "percent"]
    out["percent"] = out["percent"].round(1)
    return out


def presentation_status(df: pd.DataFrame, group_col: str = None) -> pd.DataFrame:
    """First presentation vs follow-up counts."""
    if group_col:
        counts = df.groupby(group_col)["presentation"].value_counts()
        perc = df.groupby(group_col)["presentation"].value_counts(normalize=True) * 100
    else:
        counts = df["presentation"].value_counts()
        perc = df["presentation"].value_counts(normalize=True) * 100

    out = pd.concat([counts, perc], axis=1)
    out.columns = ["N", "percent"]
    out["percent"] = out["percent"].round(1)
    return out



def main():
    df = load_data("../data/anonymized_dataset/Tumorboard_ChatGPT_anomyzed_dataset.csv")

    # Overall stats
    age_overall = age_statistics(df)
    gender_overall = gender_distribution(df)
    presentation_overall = presentation_status(df)

    # By tumor subgroup
    age_by_tumour = age_statistics(df, group_col="tumour_type")
    gender_by_tumour = gender_distribution(df, group_col="tumour_type")
    presentation_by_tumour = presentation_status(df, group_col="tumour_type")

    # ---- PRINT ----
    print("\n=== Age (Overall) ===")
    print(age_overall)

    print("\n=== Age by Tumour Type ===")
    print(age_by_tumour)

    print("\n=== Gender (Overall) ===")
    print(gender_overall)

    print("\n=== Gender by Tumour Type ===")
    print(gender_by_tumour)

    print("\n=== Presentation Status (Overall) ===")
    print("=== Legend: 1=first presentation; 2= follow-up presentation ===")
    print(presentation_overall)

    print("\n=== Presentation Status by Tumour Type ===")
    print("=== Legend: 1=first presentation; 2= follow-up presentation ===")
    print(presentation_by_tumour)


    # Export tables for manuscript / supplementary material
    age_overall.to_csv("tables/table_age_overall.csv")
    age_by_tumour.to_csv("tables/table_age_by_tumour.csv")

    gender_overall.to_csv("tables/table_gender_overall.csv")
    gender_by_tumour.to_csv("tables/table_gender_by_tumour.csv")

    presentation_overall.to_csv("tables/table_presentation_overall.csv")
    presentation_by_tumour.to_csv("tables/table_presentation_by_tumour.csv")



if __name__ == "__main__":
    main()
