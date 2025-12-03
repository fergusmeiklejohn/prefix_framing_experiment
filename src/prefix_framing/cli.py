"""Command-line interface for prefix framing experiment."""

import argparse
import sys

from rich.console import Console

console = Console()


def run_pilot():
    """Run a pilot experiment."""
    parser = argparse.ArgumentParser(description="Run a pilot prefix framing experiment")
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
        "--prefixes-per-category", "-x",
        type=int,
        default=1,
        help="Number of prefixes per category (default: 1, covers all 5 categories)",
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
        help="Skip LLM-as-judge evaluations",
    )

    args = parser.parse_args()

    from .runner import run_pilot as _run_pilot

    num_categories = 5  # emotional_positive, emotional_negative, cognitive, epistemic, control
    total_prefixes = args.prefixes_per_category * num_categories

    console.print(f"[bold]Starting Pilot Experiment[/bold]")
    console.print(f"Model: {args.model}")
    console.print(f"Prompts: {args.prompts}")
    console.print(f"Prefixes per category: {args.prefixes_per_category}")
    console.print(f"Total prefixes (across {num_categories} categories): {total_prefixes}")
    console.print(f"Replications: {args.replications}")
    console.print(f"Total trials: {args.prompts * total_prefixes * args.replications}")
    console.print()
    console.print("[bold]Selecting prefixes from each category:[/bold]")

    try:
        experiment_id = _run_pilot(
            model=args.model,
            num_prompts=args.prompts,
            num_prefixes_per_category=args.prefixes_per_category,
            replications=args.replications,
            db_path=args.db,
            run_evaluations=not args.no_eval,
        )
        console.print(f"\n[green]Experiment completed: {experiment_id}[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def run_experiment():
    """Run the full experiment."""
    parser = argparse.ArgumentParser(description="Run full prefix framing experiment")
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

    args = parser.parse_args()

    from .models import ExperimentConfig
    from .providers.ollama import OllamaProvider
    from .runner import ExperimentRunner
    from .storage import ExperimentStorage

    config = ExperimentConfig(
        name=args.name,
        model=args.model,
        judge_model=args.model,
        replications=args.replications,
        temperature=args.temperature,
    )

    console.print(f"[bold]Starting Full Experiment[/bold]")
    console.print(f"Name: {config.name}")
    console.print(f"Model: {config.model}")
    console.print(f"Replications: {config.replications}")
    console.print(f"Temperature: {config.temperature}")
    console.print()

    try:
        provider = OllamaProvider(model=args.model)
        storage = ExperimentStorage(db_path=args.db)
        runner = ExperimentRunner(provider=provider, storage=storage, config=config)

        experiment_id = runner.run(run_evaluations=not args.no_eval)
        console.print(f"\n[green]Experiment completed: {experiment_id}[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def analyze():
    """Analyze experiment results."""
    parser = argparse.ArgumentParser(description="Analyze prefix framing experiment results")
    parser.add_argument(
        "experiment_id",
        help="Experiment ID to analyze",
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

    args = parser.parse_args()

    from .analysis import generate_report
    from .storage import ExperimentStorage

    try:
        storage = ExperimentStorage(db_path=args.db)
        generate_report(storage, args.experiment_id, output_dir=args.output)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    run_pilot()
