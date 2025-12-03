#!/usr/bin/env python3
"""
Full experiment run script for prefix framing study.

This runs the complete experiment with all prompts, prefixes, and replications.

Usage:
    python scripts/run_experiment.py
    python scripts/run_experiment.py --model mistral --replications 50
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel

from prefix_framing.data.prefixes import PREFIXES
from prefix_framing.data.prompts import PROMPTS
from prefix_framing.models import ExperimentConfig
from prefix_framing.providers.ollama import OllamaProvider
from prefix_framing.runner import ExperimentRunner
from prefix_framing.storage import ExperimentStorage


console = Console()


def main():
    """Run the full experiment."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run full prefix framing experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model", "-m",
        default="llama3.2",
        help="Ollama model name (default: llama3.2)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Model for LLM-as-judge (default: same as generation model)",
    )
    parser.add_argument(
        "--replications", "-r",
        type=int,
        default=30,
        help="Replications per condition (default: 30)",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--db",
        default="results/experiment.db",
        help="Database path (default: results/experiment.db)",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip LLM-as-judge evaluations",
    )
    parser.add_argument(
        "--name",
        default="Prefix Framing Experiment",
        help="Experiment name",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume experiment with this ID (skips completed trials)",
    )

    args = parser.parse_args()

    num_prompts = len(PROMPTS)
    num_prefixes = len(PREFIXES)
    total_trials = num_prompts * num_prefixes * args.replications

    judge_model = args.judge_model or args.model

    # Show configuration
    console.print(Panel.fit(
        f"""[bold cyan]Prefix Framing Experiment - Full Run[/bold cyan]

[bold]Configuration:[/bold]
  Name:         {args.name}
  Model:        {args.model}
  Judge Model:  {judge_model}
  Temperature:  {args.temperature}
  Prompts:      {num_prompts}
  Prefixes:     {num_prefixes}
  Replications: {args.replications}
  Total Trials: {total_trials}
  Evaluations:  {"Yes" if not args.no_eval else "No (skipped)"}
  Database:     {args.db}
  Resume ID:    {args.resume or "None (new experiment)"}

[dim]Estimated time: ~{total_trials * 5 / 60:.0f} minutes (varies by model)[/dim]
        """,
        title="Configuration",
    ))

    # Confirm
    console.print(f"\n[yellow]This will run {total_trials} trials. Continue? [y/N][/yellow] ", end="")
    response = input().strip().lower()
    if response != "y":
        console.print("[red]Aborted.[/red]")
        return

    try:
        config = ExperimentConfig(
            name=args.name,
            model=args.model,
            judge_model=judge_model,
            replications=args.replications,
            temperature=args.temperature,
        )

        if args.resume:
            config.experiment_id = args.resume

        provider = OllamaProvider(model=args.model)

        # Use separate judge provider if different model
        judge_provider = None
        if judge_model != args.model:
            judge_provider = OllamaProvider(model=judge_model)

        storage = ExperimentStorage(db_path=args.db)

        runner = ExperimentRunner(
            provider=provider,
            storage=storage,
            config=config,
            judge_provider=judge_provider,
        )

        experiment_id = runner.run(run_evaluations=not args.no_eval)

        console.print(Panel.fit(
            f"""[bold green]Experiment Complete![/bold green]

Experiment ID: {experiment_id}
Database: {args.db}

[bold]Next steps:[/bold]
1. Analyze results:
   python scripts/analyze.py {experiment_id}

2. Export to JSON:
   python -c "from prefix_framing.storage import ExperimentStorage; ExperimentStorage('{args.db}').export_to_json('{experiment_id}', 'results/export.json')"
            """,
            title="Success",
        ))

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        console.print("[dim]Partial results are saved. Resume with --resume <experiment_id>[/dim]")
        sys.exit(1)

    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
