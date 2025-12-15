"""Experiment runner for prefix framing study."""

import itertools
import random
from datetime import datetime
from typing import Callable, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from .data.prefixes import PREFIXES, get_prefix
from .data.prompts import PROMPTS, get_prompt
from .data.framings import FRAMINGS, BASE_QUESTIONS
from .metrics import detect_course_correction, extract_metrics
from .models import (
    ExperimentConfig,
    FramingCategory,
    JudgeRatings,
    PrefixCategory,
    Prefix,
    Prompt,
    PromptFraming,
    PromptType,
    Trial,
)
from .providers.base import LLMProvider
from .storage import ExperimentStorage


console = Console()


class ExperimentRunner:
    """Runs the prefix framing experiment."""

    def __init__(
        self,
        provider: LLMProvider,
        storage: ExperimentStorage,
        config: ExperimentConfig,
        judge_provider: Optional[LLMProvider] = None,
    ):
        """
        Initialize the experiment runner.

        Args:
            provider: LLM provider for generating responses
            storage: Storage backend for results
            config: Experiment configuration
            judge_provider: Optional separate provider for LLM-as-judge (defaults to same)
        """
        self.provider = provider
        self.storage = storage
        self.config = config
        self.judge_provider = judge_provider or provider

    def get_prompts(self) -> list[Prompt]:
        """Get prompts to use based on config."""
        if self.config.prompt_ids:
            return [get_prompt(pid) for pid in self.config.prompt_ids]
        return PROMPTS

    def get_prefixes(self) -> list[Prefix]:
        """Get prefixes to use based on config."""
        if self.config.prefix_ids:
            return [get_prefix(pid) for pid in self.config.prefix_ids]
        return PREFIXES

    def generate_trial(
        self,
        prompt: Prompt,
        prefix: Prefix,
        replication: int,
    ) -> Trial:
        """Generate a single trial."""
        # Generate response with prefix
        result = self.provider.generate_with_prefix(
            prompt=prompt.text,
            prefix=prefix.text,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # Check for course correction
        course_corrected = detect_course_correction(prefix.text, result.continuation)

        # Create trial object
        trial = Trial(
            timestamp=datetime.now(),
            model=self.config.model,
            prompt_id=prompt.id,
            prompt_text=prompt.text,
            prompt_type=prompt.type,
            prefix_id=prefix.id,
            prefix_text=prefix.text,
            prefix_category=prefix.category,
            replication=replication,
            temperature=self.config.temperature,
            full_response=result.full_response,
            continuation_only=result.continuation,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            generation_time_ms=result.generation_time_ms,
            course_corrected=course_corrected,
        )

        # Extract metrics
        trial.metrics = extract_metrics(result.full_response)

        return trial

    def generate_framing_trial(
        self,
        base_question: str,
        framing: PromptFraming,
        replication: int,
    ) -> Trial:
        """Generate a single trial for the framing study."""
        # Apply framing to question
        framed_prompt = framing.apply(base_question)

        # Generate response without prefix forcing (natural response)
        result = self.provider.generate(
            prompt=framed_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # Create trial object
        # Use framing_id as prompt_id for compatibility with analysis
        trial = Trial(
            timestamp=datetime.now(),
            model=self.config.model,
            prompt_id=f"{framing.id}_{hash(base_question) % 10000}",
            prompt_text=framed_prompt,
            prompt_type=PromptType.FACTUAL,  # Default; could infer from question
            prefix_id="none",
            prefix_text="",
            prefix_category=PrefixCategory.CONTROL,
            replication=replication,
            temperature=self.config.temperature,
            full_response=result.full_response,
            continuation_only=result.full_response,  # No prefix, so same as full
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            generation_time_ms=result.generation_time_ms,
            course_corrected=False,
            # Framing-specific fields
            framing_id=framing.id,
            framing_category=framing.category,
            base_question=base_question,
        )

        # Extract metrics
        trial.metrics = extract_metrics(result.full_response)

        return trial

    def run_framing_study(
        self,
        questions: list[str] = None,
        framings: list[PromptFraming] = None,
        run_evaluations: bool = True,
        shuffle: bool = True,
        callback: Optional[Callable[[Trial, int, int], None]] = None,
    ) -> str:
        """
        Run the prompt-side framing study.

        Args:
            questions: Base questions to use (defaults to BASE_QUESTIONS)
            framings: Framings to apply (defaults to FRAMINGS)
            run_evaluations: Whether to run LLM-as-judge evaluations
            shuffle: Whether to randomize trial order
            callback: Optional callback(trial, current, total) after each trial

        Returns:
            experiment_id
        """
        questions = questions or BASE_QUESTIONS
        framings = framings or FRAMINGS

        # Save experiment config
        experiment_id = self.storage.save_experiment(self.config)

        replications = range(1, self.config.replications + 1)

        # Generate all conditions
        conditions = list(itertools.product(questions, framings, replications))
        total_conditions = len(conditions)

        # Check for already completed conditions (for resuming)
        completed = self.storage.get_completed_framing_conditions(experiment_id)
        conditions = [
            (q, f, r) for q, f, r in conditions
            if (q, f.id, r) not in completed
        ]

        if shuffle:
            random.shuffle(conditions)

        remaining = len(conditions)
        console.print(f"\n[bold]Starting framing study: {self.config.name}[/bold]")
        console.print(f"Model: {self.config.model}")
        console.print(f"Questions: {len(questions)}")
        console.print(f"Framings: {[f.id for f in framings]}")
        console.print(f"Replications: {self.config.replications}")
        console.print(f"Total conditions: {total_conditions}")
        console.print(f"Already completed: {total_conditions - remaining}")
        console.print(f"Remaining: {remaining}")
        console.print()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running framing trials...", total=remaining)

            for i, (question, framing, replication) in enumerate(conditions):
                try:
                    # Generate trial
                    trial = self.generate_framing_trial(question, framing, replication)

                    # Run evaluation if requested
                    if run_evaluations:
                        trial.judge_ratings = self.evaluate_trial(trial)

                    # Save trial
                    self.storage.save_trial(experiment_id, trial)

                    # Callback
                    if callback:
                        callback(trial, i + 1, remaining)

                except Exception as e:
                    console.print(
                        f"[red]Error in trial {framing.id}/{replication}: {e}[/red]"
                    )
                    # Continue with next trial

                progress.update(task, advance=1)

        # Update end time
        self.storage.update_experiment_end_time(experiment_id)

        # Print summary
        summary = self.storage.get_experiment_summary(experiment_id)
        if summary:
            self._print_summary(summary)

        return experiment_id

    def evaluate_trial(self, trial: Trial) -> JudgeRatings:
        """Evaluate a trial using LLM-as-judge."""
        # Use continuation only (without prefix) to avoid bias
        response_to_evaluate = trial.continuation_only

        ratings_dict = self.judge_provider.evaluate_response(
            question=trial.prompt_text,
            response=response_to_evaluate,
            temperature=0.0,  # Deterministic evaluation
        )

        return JudgeRatings(**ratings_dict)

    def run(
        self,
        run_evaluations: bool = True,
        shuffle: bool = True,
        callback: Optional[Callable[[Trial, int, int], None]] = None,
    ) -> str:
        """
        Run the full experiment.

        Args:
            run_evaluations: Whether to run LLM-as-judge evaluations
            shuffle: Whether to randomize trial order
            callback: Optional callback(trial, current, total) after each trial

        Returns:
            experiment_id
        """
        # Save experiment config
        experiment_id = self.storage.save_experiment(self.config)

        prompts = self.get_prompts()
        prefixes = self.get_prefixes()
        replications = range(1, self.config.replications + 1)

        # Generate all conditions
        conditions = list(itertools.product(prompts, prefixes, replications))
        total_conditions = len(conditions)

        # Check for already completed conditions (for resuming)
        completed = self.storage.get_completed_conditions(experiment_id)
        conditions = [
            (p, x, r) for p, x, r in conditions
            if (p.id, x.id, r) not in completed
        ]

        if shuffle:
            random.shuffle(conditions)

        remaining = len(conditions)
        console.print(f"\n[bold]Starting experiment: {self.config.name}[/bold]")
        console.print(f"Model: {self.config.model}")
        console.print(f"Total conditions: {total_conditions}")
        console.print(f"Already completed: {total_conditions - remaining}")
        console.print(f"Remaining: {remaining}")
        console.print()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running trials...", total=remaining)

            for i, (prompt, prefix, replication) in enumerate(conditions):
                try:
                    # Generate trial
                    trial = self.generate_trial(prompt, prefix, replication)

                    # Run evaluation if requested
                    if run_evaluations:
                        trial.judge_ratings = self.evaluate_trial(trial)

                    # Save trial
                    self.storage.save_trial(experiment_id, trial)

                    # Callback
                    if callback:
                        callback(trial, i + 1, remaining)

                except Exception as e:
                    console.print(f"[red]Error in trial {prompt.id}/{prefix.id}/{replication}: {e}[/red]")
                    # Continue with next trial

                progress.update(task, advance=1)

        # Update end time
        self.storage.update_experiment_end_time(experiment_id)

        # Print summary
        summary = self.storage.get_experiment_summary(experiment_id)
        if summary:
            self._print_summary(summary)

        return experiment_id

    def _print_summary(self, summary):
        """Print experiment summary."""
        console.print("\n[bold green]Experiment Complete![/bold green]\n")

        table = Table(title="Experiment Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Experiment ID", summary.experiment_id)
        table.add_row("Total Trials", str(summary.total_trials))
        table.add_row("Input Tokens", f"{summary.total_input_tokens:,}")
        table.add_row("Output Tokens", f"{summary.total_output_tokens:,}")
        table.add_row("Total Tokens", f"{summary.total_input_tokens + summary.total_output_tokens:,}")

        if summary.start_time and summary.end_time:
            duration = summary.end_time - summary.start_time
            table.add_row("Duration", str(duration))

        console.print(table)

        # Cost estimates
        console.print("\n[bold]Estimated API Costs (if using paid APIs):[/bold]")
        total_tokens = summary.total_input_tokens + summary.total_output_tokens

        # Rough estimates based on typical pricing
        estimates = [
            ("Claude Sonnet", 0.003, 0.015),  # per 1K input/output
            ("Claude Opus", 0.015, 0.075),
            ("GPT-4o", 0.005, 0.015),
            ("GPT-4o-mini", 0.00015, 0.0006),
        ]

        cost_table = Table()
        cost_table.add_column("Model", style="cyan")
        cost_table.add_column("Est. Cost", style="green")

        for model_name, input_rate, output_rate in estimates:
            cost = (
                (summary.total_input_tokens / 1000) * input_rate +
                (summary.total_output_tokens / 1000) * output_rate
            )
            cost_table.add_row(model_name, f"${cost:.2f}")

        console.print(cost_table)


def run_pilot(
    model: str = "llama3.2",
    num_prompts: int = 5,
    num_prefixes_per_category: int = 1,
    replications: int = 10,
    db_path: str = "results/pilot.db",
    run_evaluations: bool = True,
) -> str:
    """
    Run a small pilot experiment.

    Selects prefixes from ALL categories to ensure proper experimental design
    with control conditions for valid comparisons.

    Args:
        model: Ollama model name
        num_prompts: Number of prompts to use
        num_prefixes_per_category: Number of prefixes per category (default: 1)
        replications: Replications per condition
        db_path: Path to database file
        run_evaluations: Whether to run LLM-as-judge

    Returns:
        experiment_id
    """
    from .providers.ollama import OllamaProvider
    from .models import PrefixCategory

    # Select subset of prompts
    prompt_ids = [p.id for p in PROMPTS[:num_prompts]]

    # Select prefixes from EACH category to ensure balanced design with control
    prefix_ids = []
    for category in PrefixCategory:
        category_prefixes = [p for p in PREFIXES if p.category == category]
        selected = category_prefixes[:num_prefixes_per_category]
        prefix_ids.extend([p.id for p in selected])
        console.print(f"  {category.value}: {[p.id for p in selected]}")

    console.print(f"\n[bold]Selected {len(prefix_ids)} prefixes across all categories[/bold]")

    config = ExperimentConfig(
        name="Pilot Experiment",
        model=model,
        judge_model=model,
        replications=replications,
        prompt_ids=prompt_ids,
        prefix_ids=prefix_ids,
    )

    provider = OllamaProvider(model=model)
    storage = ExperimentStorage(db_path=db_path)

    runner = ExperimentRunner(
        provider=provider,
        storage=storage,
        config=config,
    )

    return runner.run(run_evaluations=run_evaluations)


def run_framing_study(
    model: str = "llama3.2",
    questions: list[str] = None,
    replications: int = 30,
    db_path: str = "results/framing_study.db",
    run_evaluations: bool = True,
) -> str:
    """
    Run the prompt-side framing study.

    Tests whether user engagement framing affects response quality.

    Args:
        model: Ollama model name
        questions: Base questions to use (defaults to BASE_QUESTIONS)
        replications: Replications per condition
        db_path: Path to database file
        run_evaluations: Whether to run LLM-as-judge

    Returns:
        experiment_id
    """
    from .providers.ollama import OllamaProvider

    questions = questions or BASE_QUESTIONS

    console.print("\n[bold]Prompt-Side Framing Study[/bold]")
    console.print(f"Testing whether user engagement framing affects response quality\n")
    console.print(f"Framings: {[f.id for f in FRAMINGS]}")
    console.print(f"Questions: {len(questions)}")
    console.print(f"Replications: {replications}")
    console.print(f"Total trials: {len(questions) * len(FRAMINGS) * replications}")

    config = ExperimentConfig(
        name="Framing Study",
        model=model,
        judge_model=model,
        replications=replications,
    )

    provider = OllamaProvider(model=model)
    storage = ExperimentStorage(db_path=db_path)

    runner = ExperimentRunner(
        provider=provider,
        storage=storage,
        config=config,
    )

    return runner.run_framing_study(
        questions=questions,
        run_evaluations=run_evaluations,
    )
