#!/usr/bin/env python3
"""
Analysis script for framing study results.

Usage:
    python scripts/analyze_framing.py <experiment_id> --db results/framing_study.db
    python scripts/analyze_framing.py --db results/framing_study.db  # List experiments
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.table import Table

from prefix_framing.storage import ExperimentStorage
from prefix_framing.analysis import generate_framing_report


console = Console()


def list_experiments(storage: ExperimentStorage):
    """List all experiments in the database."""
    with storage._get_conn() as conn:
        rows = conn.execute("""
            SELECT
                e.experiment_id,
                e.name,
                e.start_time,
                e.end_time,
                COUNT(t.trial_id) as trial_count,
                COUNT(DISTINCT t.framing_id) as framing_count
            FROM experiments e
            LEFT JOIN trials t ON e.experiment_id = t.experiment_id
            GROUP BY e.experiment_id
            ORDER BY e.start_time DESC
        """).fetchall()

    if not rows:
        console.print("[yellow]No experiments found in database.[/yellow]")
        return

    table = Table(title="Experiments in Database")
    table.add_column("Experiment ID", style="cyan")
    table.add_column("Name")
    table.add_column("Trials", justify="right")
    table.add_column("Framings", justify="right")
    table.add_column("Start Time")

    for row in rows:
        framing_note = f"{row['framing_count']}" if row['framing_count'] > 0 else "[dim]N/A[/dim]"
        table.add_row(
            row["experiment_id"][:8] + "...",
            row["name"] or "N/A",
            str(row["trial_count"]),
            framing_note,
            row["start_time"][:19] if row["start_time"] else "N/A",
        )

    console.print(table)
    console.print("\n[dim]Use: python scripts/analyze_framing.py <experiment_id> --db <db_path>[/dim]")


def main():
    """Run framing study analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze framing study results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "experiment_id",
        nargs="?",
        help="Experiment ID to analyze (omit to list experiments)",
    )
    parser.add_argument(
        "--db",
        default="results/framing_study.db",
        help="Database path (default: results/framing_study.db)",
    )
    parser.add_argument(
        "--output", "-o",
        default="results/framing_analysis",
        help="Output directory for analysis files",
    )

    args = parser.parse_args()

    # Check database exists
    if not Path(args.db).exists():
        console.print(f"[red]Database not found: {args.db}[/red]")
        console.print("[dim]Run a framing study first: python scripts/run_framing_study.py[/dim]")
        sys.exit(1)

    storage = ExperimentStorage(db_path=args.db)

    if not args.experiment_id:
        list_experiments(storage)
        return

    # Run analysis
    console.print(f"[bold]Analyzing framing study: {args.experiment_id}[/bold]\n")

    try:
        generate_framing_report(
            storage=storage,
            experiment_id=args.experiment_id,
            output_dir=args.output,
        )
    except Exception as e:
        console.print(f"[red]Error during analysis: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
