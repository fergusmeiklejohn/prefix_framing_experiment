"""Analysis and visualization for prefix framing experiment."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from .models import FramingCategory, PrefixCategory, PromptType
from .storage import ExperimentStorage


def trials_to_dataframe(storage: ExperimentStorage, experiment_id: str) -> pd.DataFrame:
    """
    Convert experiment trials to a pandas DataFrame for analysis.

    Args:
        storage: Experiment storage
        experiment_id: ID of experiment to analyze

    Returns:
        DataFrame with one row per trial
    """
    trials = list(storage.get_trials(experiment_id=experiment_id))

    rows = []
    for trial in trials:
        row = {
            "trial_id": trial.trial_id,
            "prompt_id": trial.prompt_id,
            "prompt_type": trial.prompt_type.value,
            "prefix_id": trial.prefix_id,
            "prefix_category": trial.prefix_category.value,
            "prefix_text": trial.prefix_text,
            "replication": trial.replication,
            "temperature": trial.temperature,
            "input_tokens": trial.input_tokens,
            "output_tokens": trial.output_tokens,
            "generation_time_ms": trial.generation_time_ms,
            "course_corrected": trial.course_corrected,
            # Framing study fields
            "framing_id": trial.framing_id,
            "framing_category": trial.framing_category.value if trial.framing_category else None,
            "base_question": trial.base_question,
        }

        # Add metrics
        if trial.metrics:
            for field, value in trial.metrics.model_dump().items():
                row[f"m_{field}"] = value

        # Add judge ratings
        if trial.judge_ratings:
            for field, value in trial.judge_ratings.model_dump().items():
                if field != "brief_justification":
                    row[f"j_{field}"] = value

        rows.append(row)

    return pd.DataFrame(rows)


def compute_summary_stats(df: pd.DataFrame, group_by: str = "prefix_category") -> pd.DataFrame:
    """
    Compute summary statistics grouped by a column.

    Args:
        df: Trial DataFrame
        group_by: Column to group by

    Returns:
        DataFrame with mean, std, count for each metric
    """
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["replication", "temperature"]
    metric_cols = [c for c in numeric_cols if c not in exclude_cols]

    # Compute grouped stats
    stats_df = df.groupby(group_by)[metric_cols].agg(["mean", "std", "count"])

    return stats_df


def compute_effect_sizes(
    df: pd.DataFrame,
    baseline_category: str = "control",
    metric_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute Cohen's d effect sizes comparing each category to baseline.

    Args:
        df: Trial DataFrame
        baseline_category: Category to use as baseline
        metric_cols: Columns to compute effect sizes for (default: all metrics)

    Returns:
        DataFrame with effect sizes
    """
    if metric_cols is None:
        metric_cols = [c for c in df.columns if c.startswith("m_") or c.startswith("j_")]

    categories = df["prefix_category"].unique()

    # Check if baseline exists
    if baseline_category not in categories:
        print(f"\n[WARNING] No '{baseline_category}' category found in data!")
        print(f"  Available categories: {list(categories)}")
        print(f"  Effect sizes cannot be computed without a baseline.")
        print(f"  Consider re-running the experiment with control prefixes (D1, D2).\n")
        return pd.DataFrame(columns=["category"])

    baseline = df[df["prefix_category"] == baseline_category]

    if len(baseline) < 2:
        print(f"\n[WARNING] Baseline category '{baseline_category}' has only {len(baseline)} trial(s).")
        print(f"  Need at least 2 trials for effect size calculation.\n")
        return pd.DataFrame(columns=["category"])

    results = []
    for category in categories:
        if category == baseline_category:
            continue

        treatment = df[df["prefix_category"] == category]

        row = {"category": category}
        for col in metric_cols:
            if col in baseline.columns and col in treatment.columns:
                baseline_vals = baseline[col].dropna()
                treatment_vals = treatment[col].dropna()

                if len(baseline_vals) > 1 and len(treatment_vals) > 1:
                    # Cohen's d
                    pooled_std = np.sqrt(
                        ((len(baseline_vals) - 1) * baseline_vals.std() ** 2 +
                         (len(treatment_vals) - 1) * treatment_vals.std() ** 2) /
                        (len(baseline_vals) + len(treatment_vals) - 2)
                    )
                    if pooled_std > 0:
                        d = (treatment_vals.mean() - baseline_vals.mean()) / pooled_std
                    else:
                        d = 0
                    row[col] = round(d, 3)

        results.append(row)

    return pd.DataFrame(results)


def run_anova(
    df: pd.DataFrame,
    metric: str,
    group_col: str = "prefix_category",
) -> dict:
    """
    Run one-way ANOVA for a metric across groups.

    Args:
        df: Trial DataFrame
        metric: Metric column name
        group_col: Column to group by

    Returns:
        Dict with F-statistic, p-value, and group means
    """
    groups = df.groupby(group_col)[metric].apply(list).to_dict()
    group_arrays = [np.array(v) for v in groups.values() if len(v) > 0]

    if len(group_arrays) < 2:
        return {"error": "Not enough groups for ANOVA"}

    f_stat, p_value = stats.f_oneway(*group_arrays)

    return {
        "metric": metric,
        "f_statistic": round(f_stat, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < 0.05,
        "group_means": {k: round(np.mean(v), 3) for k, v in groups.items()},
    }


def plot_metric_by_category(
    df: pd.DataFrame,
    metric: str,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 6),
):
    """
    Create a box plot of a metric by prefix category.

    Args:
        df: Trial DataFrame
        metric: Metric column to plot
        output_path: Optional path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)

    # Order categories
    category_order = [
        "control",
        "emotional_positive",
        "emotional_negative",
        "cognitive",
        "epistemic",
    ]
    category_order = [c for c in category_order if c in df["prefix_category"].unique()]

    sns.boxplot(
        data=df,
        x="prefix_category",
        y=metric,
        order=category_order,
        hue="prefix_category",
        palette="Set2",
        legend=False,
    )

    plt.title(f"{metric} by Prefix Category")
    plt.xlabel("Prefix Category")
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_effect_size_heatmap(
    effect_sizes_df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: tuple = (12, 8),
):
    """
    Create a heatmap of effect sizes.

    Args:
        effect_sizes_df: DataFrame from compute_effect_sizes
        output_path: Optional path to save figure
        figsize: Figure size
    """
    # Pivot for heatmap
    df_plot = effect_sizes_df.set_index("category")

    plt.figure(figsize=figsize)
    sns.heatmap(
        df_plot,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Cohen's d"},
    )

    plt.title("Effect Sizes by Category (vs Control)")
    plt.xlabel("Metric")
    plt.ylabel("Prefix Category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_judge_ratings_radar(
    df: pd.DataFrame,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 10),
):
    """
    Create a radar chart of judge ratings by prefix category.

    Args:
        df: Trial DataFrame
        output_path: Optional path to save figure
        figsize: Figure size
    """
    rating_cols = [c for c in df.columns if c.startswith("j_") and c != "j_brief_justification"]

    if not rating_cols:
        print("No judge ratings found in data")
        return

    # Compute means by category
    means = df.groupby("prefix_category")[rating_cols].mean()

    # Set up radar chart
    categories = [c.replace("j_", "") for c in rating_cols]
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    colors = plt.cm.Set2(np.linspace(0, 1, len(means)))

    for (category, values), color in zip(means.iterrows(), colors):
        vals = values.tolist()
        vals += vals[:1]  # Close the loop
        ax.plot(angles, vals, "o-", linewidth=2, label=category, color=color)
        ax.fill(angles, vals, alpha=0.25, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(1, 7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.title("Judge Ratings by Prefix Category")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def generate_report(
    storage: ExperimentStorage,
    experiment_id: str,
    output_dir: str = "results/analysis",
):
    """
    Generate a full analysis report with plots.

    Args:
        storage: Experiment storage
        experiment_id: Experiment to analyze
        output_dir: Directory for output files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    df = trials_to_dataframe(storage, experiment_id)

    if df.empty:
        print("No trials found for experiment")
        return

    print(f"Loaded {len(df)} trials")

    # Validate experimental design
    categories = df["prefix_category"].unique()
    print(f"Prefix categories in data: {list(categories)}")

    if "control" not in categories:
        print("\n" + "="*60)
        print("[WARNING] NO CONTROL CONDITION FOUND")
        print("="*60)
        print("Effect sizes require a control/baseline condition.")
        print("Your experiment only includes:", list(categories))
        print("="*60 + "\n")

    # Check for judge rating issues
    judge_cols = [c for c in df.columns if c.startswith("j_") and c != "j_brief_justification"]
    if judge_cols:
        # Check if all ratings are identical (indicates evaluation failure)
        all_same = True
        for col in judge_cols:
            if df[col].nunique() > 1:
                all_same = False
                break

        if all_same and len(df) > 10:
            print("\n" + "="*60)
            print("[WARNING] ALL JUDGE RATINGS ARE IDENTICAL")
            print("="*60)
            print("This usually indicates the LLM-as-judge evaluation failed.")
            print("All ratings defaulted to the same value (likely 4).")
            print("Judge ratings in this analysis are NOT meaningful.")
            print("Consider: ")
            print("  - Using a different model for evaluation")
            print("  - Running with PF_OLLAMA_DEBUG=1 to diagnose")
            print("  - Re-running with --no-eval and manual evaluation")
            print("="*60 + "\n")

    # Summary stats
    summary = compute_summary_stats(df)
    summary.to_csv(output_path / "summary_stats.csv")
    print(f"Saved summary stats to {output_path / 'summary_stats.csv'}")

    # Effect sizes
    effect_sizes = compute_effect_sizes(df)
    effect_sizes.to_csv(output_path / "effect_sizes.csv", index=False)
    print(f"Saved effect sizes to {output_path / 'effect_sizes.csv'}")

    # ANOVA for key metrics
    key_metrics = ["m_word_count", "m_hedge_word_count", "j_thoroughness", "j_engagement"]
    available_metrics = [m for m in key_metrics if m in df.columns]

    anova_results = []
    for metric in available_metrics:
        result = run_anova(df, metric)
        anova_results.append(result)
        print(f"ANOVA for {metric}: F={result.get('f_statistic', 'N/A')}, p={result.get('p_value', 'N/A')}")

    # Save ANOVA results
    pd.DataFrame(anova_results).to_csv(output_path / "anova_results.csv", index=False)

    # Generate plots
    for metric in ["m_word_count", "m_hedge_word_count"]:
        if metric in df.columns:
            plot_metric_by_category(
                df, metric,
                output_path=str(output_path / f"boxplot_{metric}.png")
            )
            print(f"Saved boxplot for {metric}")

    # Effect size heatmap
    # Check that effect_sizes has data columns (not just 'category')
    if not effect_sizes.empty and len(effect_sizes.columns) > 1:
        plot_effect_size_heatmap(
            effect_sizes,
            output_path=str(output_path / "effect_size_heatmap.png")
        )
        print("Saved effect size heatmap")
    else:
        print("Skipping effect size heatmap (no baseline 'control' category found or no effect sizes computed)")

    # Radar chart for judge ratings
    plot_judge_ratings_radar(
        df,
        output_path=str(output_path / "judge_ratings_radar.png")
    )
    print("Saved judge ratings radar chart")

    # Export full data
    df.to_csv(output_path / "full_data.csv", index=False)
    print(f"Saved full data to {output_path / 'full_data.csv'}")

    print(f"\nAnalysis complete! Results in {output_path}")


# ============================================================================
# Framing Study Analysis Functions
# ============================================================================


def compute_framing_effect_sizes(
    df: pd.DataFrame,
    baseline_category: str = "neutral",
    metric_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute Cohen's d effect sizes comparing each framing to baseline (neutral).

    Args:
        df: Trial DataFrame with framing_category column
        baseline_category: Category to use as baseline (default: neutral)
        metric_cols: Columns to compute effect sizes for (default: all metrics)

    Returns:
        DataFrame with effect sizes
    """
    if "framing_category" not in df.columns or df["framing_category"].isna().all():
        print("[WARNING] No framing_category data found. Is this a framing study?")
        return pd.DataFrame(columns=["category"])

    if metric_cols is None:
        metric_cols = [c for c in df.columns if c.startswith("m_") or c.startswith("j_")]

    categories = df["framing_category"].dropna().unique()

    if baseline_category not in categories:
        print(f"\n[WARNING] No '{baseline_category}' framing found in data!")
        print(f"  Available framings: {list(categories)}")
        return pd.DataFrame(columns=["category"])

    baseline = df[df["framing_category"] == baseline_category]

    if len(baseline) < 2:
        print(f"\n[WARNING] Baseline framing '{baseline_category}' has only {len(baseline)} trial(s).")
        return pd.DataFrame(columns=["category"])

    results = []
    for category in categories:
        if category == baseline_category:
            continue

        treatment = df[df["framing_category"] == category]

        row = {"category": category}
        for col in metric_cols:
            if col in baseline.columns and col in treatment.columns:
                baseline_vals = baseline[col].dropna()
                treatment_vals = treatment[col].dropna()

                if len(baseline_vals) > 1 and len(treatment_vals) > 1:
                    pooled_std = np.sqrt(
                        ((len(baseline_vals) - 1) * baseline_vals.std() ** 2 +
                         (len(treatment_vals) - 1) * treatment_vals.std() ** 2) /
                        (len(baseline_vals) + len(treatment_vals) - 2)
                    )
                    if pooled_std > 0:
                        d = (treatment_vals.mean() - baseline_vals.mean()) / pooled_std
                    else:
                        d = 0
                    row[col] = round(d, 3)

        results.append(row)

    return pd.DataFrame(results)


def plot_metric_by_framing(
    df: pd.DataFrame,
    metric: str,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 6),
):
    """
    Create a box plot of a metric by framing category.

    Args:
        df: Trial DataFrame with framing_category
        metric: Metric column to plot
        output_path: Optional path to save figure
        figsize: Figure size
    """
    if "framing_category" not in df.columns or df["framing_category"].isna().all():
        print("No framing_category data found")
        return

    plt.figure(figsize=figsize)

    # Order categories in predicted quality order
    category_order = ["enthusiastic", "neutral", "rushed", "dismissive"]
    category_order = [c for c in category_order if c in df["framing_category"].unique()]

    sns.boxplot(
        data=df,
        x="framing_category",
        y=metric,
        order=category_order,
        hue="framing_category",
        palette="coolwarm",
        legend=False,
    )

    plt.title(f"{metric} by User Framing")
    plt.xlabel("User Framing Category")
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def generate_framing_report(
    storage: ExperimentStorage,
    experiment_id: str,
    output_dir: str = "results/framing_analysis",
):
    """
    Generate analysis report for a framing study.

    Args:
        storage: Experiment storage
        experiment_id: Experiment to analyze
        output_dir: Directory for output files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    df = trials_to_dataframe(storage, experiment_id)

    if df.empty:
        print("No trials found for experiment")
        return

    # Check if this is a framing study
    if df["framing_category"].isna().all():
        print("This doesn't appear to be a framing study (no framing_category data)")
        print("Use generate_report() for prefix studies instead")
        return

    print(f"Loaded {len(df)} framing study trials")

    # Show framing distribution
    categories = df["framing_category"].dropna().unique()
    print(f"Framing categories: {list(categories)}")
    print(f"Trials per framing:")
    print(df["framing_category"].value_counts())

    # Summary stats by framing
    summary = compute_summary_stats(df, group_by="framing_category")
    summary.to_csv(output_path / "framing_summary_stats.csv")
    print(f"\nSaved summary stats to {output_path / 'framing_summary_stats.csv'}")

    # Effect sizes vs neutral
    effect_sizes = compute_framing_effect_sizes(df)
    effect_sizes.to_csv(output_path / "framing_effect_sizes.csv", index=False)
    print(f"Saved effect sizes to {output_path / 'framing_effect_sizes.csv'}")

    # ANOVA for key metrics
    key_metrics = ["m_word_count", "m_hedge_word_count", "j_thoroughness", "j_engagement"]
    available_metrics = [m for m in key_metrics if m in df.columns]

    anova_results = []
    for metric in available_metrics:
        result = run_anova(df, metric, group_col="framing_category")
        anova_results.append(result)
        print(f"ANOVA for {metric}: F={result.get('f_statistic', 'N/A')}, p={result.get('p_value', 'N/A')}")

    pd.DataFrame(anova_results).to_csv(output_path / "framing_anova_results.csv", index=False)

    # Generate plots
    for metric in ["m_word_count", "m_hedge_word_count"]:
        if metric in df.columns:
            plot_metric_by_framing(
                df, metric,
                output_path=str(output_path / f"framing_boxplot_{metric}.png")
            )
            print(f"Saved boxplot for {metric}")

    # Effect size heatmap
    if not effect_sizes.empty and len(effect_sizes.columns) > 1:
        plot_effect_size_heatmap(
            effect_sizes,
            output_path=str(output_path / "framing_effect_size_heatmap.png")
        )
        print("Saved effect size heatmap")

    # Radar chart for judge ratings
    if any(c.startswith("j_") for c in df.columns):
        # Temporarily rename framing_category to prefix_category for the existing radar function
        df_radar = df.copy()
        df_radar["prefix_category"] = df_radar["framing_category"]
        plot_judge_ratings_radar(
            df_radar,
            output_path=str(output_path / "framing_judge_ratings_radar.png")
        )
        print("Saved judge ratings radar chart")

    # Export full data
    df.to_csv(output_path / "framing_full_data.csv", index=False)
    print(f"Saved full data to {output_path / 'framing_full_data.csv'}")

    # Print key findings summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS SUMMARY")
    print("=" * 60)

    # Compare means across framings
    if "m_word_count" in df.columns:
        means = df.groupby("framing_category")["m_word_count"].mean()
        print(f"\nMean word count by framing:")
        for cat in ["enthusiastic", "neutral", "rushed", "dismissive"]:
            if cat in means.index:
                print(f"  {cat}: {means[cat]:.1f}")

    if available_metrics:
        print(f"\nSignificant ANOVA results (p < 0.05):")
        for result in anova_results:
            if result.get("significant"):
                print(f"  {result['metric']}: F={result['f_statistic']}, p={result['p_value']}")

    print("=" * 60)
    print(f"\nAnalysis complete! Results in {output_path}")
