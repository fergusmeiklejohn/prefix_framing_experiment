#!/usr/bin/env python3
"""
Pilot run script for prefix framing experiment.

This script runs a small-scale pilot to:
1. Verify Ollama is working correctly
2. Test the experimental pipeline
3. Get initial token usage estimates
4. Check for any issues before full run

Usage:
    python scripts/run_pilot.py
    python scripts/run_pilot.py --model mistral
    python scripts/run_pilot.py --prompts 3 --prefixes 3 --replications 5
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel

from prefix_framing.runner import run_pilot


console = Console()


def main():
    """Run the pilot experiment."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a pilot prefix framing experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default pilot (5 prompts × 5 prefixes × 10 reps = 250 trials)
    python scripts/run_pilot.py

    # Quick test (3 × 3 × 3 = 27 trials)
    python scripts/run_pilot.py -p 3 -x 3 -r 3

    # Use mistral model
    python scripts/run_pilot.py --model mistral

    # Skip evaluations for faster run
    python scripts/run_pilot.py --no-eval
        """,
    )

    parser.add_argument(
        "--model", "-m",
        default="llama3.2",
        help="Ollama model name (default: llama3.2)",
    )
    parser.add_argument(
        "--prompts", "-p",
        type=int,
        default=5,
        help="Number of prompts to use (default: 5)",
    )
    parser.add_argument(
        "--prefixes", "-x",
        type=int,
        default=5,
        help="Number of prefixes to use (default: 5)",
    )
    parser.add_argument(
        "--replications", "-r",
        type=int,
        default=10,
        help="Replications per condition (default: 10)",
    )
    parser.add_argument(
        "--db",
        default="results/pilot.db",
        help="Database path (default: results/pilot.db)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip LLM-as-judge evaluations (faster)",
    )

    args = parser.parse_args()

    total_trials = args.prompts * args.prefixes * args.replications

    # Show configuration
    console.print(Panel.fit(
        f"""[bold cyan]Prefix Framing Experiment - Pilot Run[/bold cyan]

[bold]Configuration:[/bold]
  Model:        {args.model}
  Prompts:      {args.prompts}
  Prefixes:     {args.prefixes}
  Replications: {args.replications}
  Total Trials: {total_trials}
  Evaluations:  {"Yes" if not args.no_eval else "No (skipped)"}
  Database:     {args.db}

[dim]Make sure Ollama is running: ollama serve[/dim]
        """,
        title="Configuration",
    ))

    # Confirm
    if total_trials > 50:
        console.print(f"\n[yellow]This will run {total_trials} trials. Continue? [y/N][/yellow] ", end="")
        response = input().strip().lower()
        if response != "y":
            console.print("[red]Aborted.[/red]")
            return

    try:
        experiment_id = run_pilot(
            model=args.model,
            num_prompts=args.prompts,
            num_prefixes=args.prefixes,
            replications=args.replications,
            db_path=args.db,
            run_evaluations=not args.no_eval,
        )

        console.print(Panel.fit(
            f"""[bold green]Pilot Complete![/bold green]

Experiment ID: {experiment_id}
Database: {args.db}

[bold]Next steps:[/bold]
1. Review results:
   python scripts/analyze.py {experiment_id} --db {args.db}

2. If satisfied, run full experiment:
   python scripts/run_experiment.py
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
