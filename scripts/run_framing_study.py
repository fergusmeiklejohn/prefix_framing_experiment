#!/usr/bin/env python3
"""
Framing study script for testing user engagement framing effects.

This study tests whether user prompt framing affects response quality:
- Enthusiastic: "I'm really curious - [question]"
- Neutral: "[question]"
- Rushed: "Quick question - [question]"
- Dismissive: "This is probably simple, but [question]"

Usage:
    python scripts/run_framing_study.py
    python scripts/run_framing_study.py --model gpt-oss
    python scripts/run_framing_study.py -r 10 --no-eval
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel

from prefix_framing.runner import run_framing_study
from prefix_framing.data.framings import FRAMINGS, BASE_QUESTIONS


console = Console()


def main():
    """Run the framing study."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a prompt-side framing study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default (5 questions × 4 framings × 30 reps = 600 trials)
    python scripts/run_framing_study.py

    # Quick test (5 × 4 × 3 = 60 trials)
    python scripts/run_framing_study.py -r 3

    # Use gpt-oss model
    python scripts/run_framing_study.py --model gpt-oss

    # Skip evaluations for faster run
    python scripts/run_framing_study.py --no-eval
        """,
    )

    parser.add_argument(
        "--model", "-m",
        default="llama3.2",
        help="Ollama model name (default: llama3.2)",
    )
    parser.add_argument(
        "--replications", "-r",
        type=int,
        default=30,
        help="Replications per condition (default: 30)",
    )
    parser.add_argument(
        "--db",
        default="results/framing_study.db",
        help="Database path (default: results/framing_study.db)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip LLM-as-judge evaluations (faster)",
    )

    args = parser.parse_args()

    num_questions = len(BASE_QUESTIONS)
    num_framings = len(FRAMINGS)
    total_trials = num_questions * num_framings * args.replications

    # Show configuration
    console.print(Panel.fit(
        f"""[bold cyan]Prompt-Side Framing Study[/bold cyan]

[bold]Research Question:[/bold]
Does user engagement framing affect response quality?

[bold]Framings:[/bold]
  - Enthusiastic: "I'm really curious - [question]"
  - Neutral: "[question]"
  - Rushed: "Quick question - [question]"
  - Dismissive: "This is probably simple, but [question]"

[bold]Configuration:[/bold]
  Model:        {args.model}
  Questions:    {num_questions}
  Framings:     {num_framings}
  Replications: {args.replications}
  Total Trials: {total_trials}
  Evaluations:  {"Yes" if not args.no_eval else "No (skipped)"}
  Database:     {args.db}

[bold]Hypothesis:[/bold]
  Enthusiastic > Neutral > Rushed > Dismissive

[dim]Make sure Ollama is running: ollama serve[/dim]
        """,
        title="Framing Study Configuration",
    ))

    # Confirm
    if total_trials > 100:
        console.print(f"\n[yellow]This will run {total_trials} trials. Continue? [y/N][/yellow] ", end="")
        response = input().strip().lower()
        if response != "y":
            console.print("[red]Aborted.[/red]")
            return

    try:
        experiment_id = run_framing_study(
            model=args.model,
            replications=args.replications,
            db_path=args.db,
            run_evaluations=not args.no_eval,
        )

        console.print(Panel.fit(
            f"""[bold green]Framing Study Complete![/bold green]

Experiment ID: {experiment_id}
Database: {args.db}

[bold]Next steps:[/bold]
1. Analyze results:
   python scripts/analyze_framing.py {experiment_id} --db {args.db}

2. Or use the general analyze script:
   python scripts/analyze.py {experiment_id} --db {args.db}
            """,
            title="Success",
        ))

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        console.print("[dim]Partial results are saved in the database.[/dim]")
        sys.exit(1)

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        console.print("\n[dim]Troubleshooting:[/dim]")
        console.print("1. Is Ollama running? Run: ollama serve")
        console.print(f"2. Is model available? Run: ollama pull {args.model}")
        console.print("3. Check Ollama logs for more details")
        sys.exit(1)


if __name__ == "__main__":
    main()
