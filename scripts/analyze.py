#!/usr/bin/env python3
"""
Analyze experiment results.

Usage:
    python scripts/analyze.py <experiment_id>
    python scripts/analyze.py <experiment_id> --db results/pilot.db
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from prefix_framing.analysis import (
    compute_effect_sizes,
    compute_summary_stats,
    generate_report,
    run_anova,
    trials_to_dataframe,
)
from prefix_framing.storage import ExperimentStorage


console = Console()


def main():
    """Analyze experiment results."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze prefix framing experiment results")
    parser.add_argument(
        "experiment_id",
        nargs="?",
        help="Experiment ID to analyze (shows list if omitted)",
    )
    parser.add_argument(
        "--db",
        default="results/experiment.db",
        help="Database path (default: results/experiment.db)",
    )
    parser.add_argument(
        "--output", "-o",
        default="results/analysis",
        help="Output directory for analysis (default: results/analysis)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick summary only (no plots)",
    )

    args = parser.parse_args()

    try:
        storage = ExperimentStorage(db_path=args.db)
    except Exception as e:
        console.print(f"[red]Error opening database: {e}[/red]")
        console.print(f"[dim]Check that {args.db} exists[/dim]")
        sys.exit(1)

    # If no experiment ID, list available experiments
    if not args.experiment_id:
        console.print("[bold]Available experiments:[/bold]")
        console.print(f"[dim]Database: {args.db}[/dim]\n")

        # Query experiments (we need to add this method)
        import sqlite3
        conn = sqlite3.connect(args.db)
        conn.row_factory = sqlite3.Row
        experiments = conn.execute(
            "SELECT experiment_id, name, start_time, end_time FROM experiments ORDER BY start_time DESC"
        ).fetchall()
        conn.close()

        if not experiments:
            console.print("[yellow]No experiments found.[/yellow]")
            return

        table = Table()
        table.add_column("ID", style="cyan")
        table.add_column("Name")
        table.add_column("Started")
        table.add_column("Trials")

        for exp in experiments:
            trial_count = storage.get_trial_count(exp["experiment_id"])
            table.add_row(
                exp["experiment_id"][:8] + "...",
                exp["name"] or "Unnamed",
                exp["start_time"] or "Unknown",
                str(trial_count),
            )

        console.print(table)
        console.print("\n[dim]Run with experiment ID to analyze:[/dim]")
        console.print(f"  python scripts/analyze.py <experiment_id> --db {args.db}")
        return

    # Load data
    console.print(f"[bold]Analyzing experiment: {args.experiment_id[:8]}...[/bold]")
    console.print(f"[dim]Database: {args.db}[/dim]\n")

    df = trials_to_dataframe(storage, args.experiment_id)

    if df.empty:
        console.print("[red]No trials found for this experiment.[/red]")
        sys.exit(1)

    console.print(f"Loaded {len(df)} trials\n")

    # Quick summary
    console.print("[bold]Summary by Prefix Category:[/bold]")

    summary_table = Table()
    summary_table.add_column("Category")
    summary_table.add_column("Count")
    summary_table.add_column("Avg Words")
    summary_table.add_column("Avg Tokens")

    for category in df["prefix_category"].unique():
        cat_df = df[df["prefix_category"] == category]
        avg_words = cat_df["m_word_count"].mean() if "m_word_count" in cat_df else 0
        avg_tokens = cat_df["output_tokens"].mean()
        summary_table.add_row(
            category,
            str(len(cat_df)),
            f"{avg_words:.1f}",
            f"{avg_tokens:.1f}",
        )

    console.print(summary_table)

    # Token usage summary
    total_input = df["input_tokens"].sum()
    total_output = df["output_tokens"].sum()
    console.print(f"\n[bold]Token Usage:[/bold]")
    console.print(f"  Input tokens:  {total_input:,}")
    console.print(f"  Output tokens: {total_output:,}")
    console.print(f"  Total tokens:  {total_input + total_output:,}")

    # Cost estimates
    console.print(f"\n[bold]Estimated API Costs:[/bold]")
    estimates = [
        ("Claude Sonnet", 0.003, 0.015),
        ("Claude Opus", 0.015, 0.075),
        ("GPT-4o", 0.005, 0.015),
        ("GPT-4o-mini", 0.00015, 0.0006),
    ]
    for model, inp_rate, out_rate in estimates:
        cost = (total_input / 1000) * inp_rate + (total_output / 1000) * out_rate
        console.print(f"  {model}: ${cost:.2f}")

    # Judge ratings summary (if available)
    judge_cols = [c for c in df.columns if c.startswith("j_")]
    if judge_cols:
        console.print(f"\n[bold]Judge Ratings by Category:[/bold]")
        rating_cols = [c for c in judge_cols if c != "j_brief_justification"]

        for category in df["prefix_category"].unique():
            cat_df = df[df["prefix_category"] == category]
            means = cat_df[rating_cols].mean()
            console.print(f"\n  {category}:")
            for col in rating_cols:
                console.print(f"    {col.replace('j_', '')}: {means[col]:.2f}")

    # ANOVA for key metrics
    console.print(f"\n[bold]Statistical Tests (ANOVA):[/bold]")
    key_metrics = ["m_word_count", "m_hedge_word_count", "j_thoroughness", "j_engagement"]
    available = [m for m in key_metrics if m in df.columns]

    for metric in available:
        result = run_anova(df, metric)
        sig = "✓" if result.get("significant") else ""
        console.print(
            f"  {metric}: F={result.get('f_statistic', 'N/A')}, "
            f"p={result.get('p_value', 'N/A')} {sig}"
        )

    # Full report
    if not args.quick:
        console.print(f"\n[bold]Generating full report...[/bold]")
        generate_report(storage, args.experiment_id, output_dir=args.output)
        console.print(f"[green]Report saved to {args.output}/[/green]")


if __name__ == "__main__":
    main()
