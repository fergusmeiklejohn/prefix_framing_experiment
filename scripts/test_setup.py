#!/usr/bin/env python3
"""
Test script to verify the setup is working correctly.

Run this before the pilot to ensure Ollama is accessible and the code works.

Usage:
    python scripts/test_setup.py
    python scripts/test_setup.py --model mistral
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel

console = Console()


def test_imports():
    """Test that all modules import correctly."""
    console.print("Testing imports...", end=" ")
    try:
        from prefix_framing.models import ExperimentConfig, Trial, Prompt, Prefix
        from prefix_framing.data.prompts import PROMPTS
        from prefix_framing.data.prefixes import PREFIXES
        from prefix_framing.storage import ExperimentStorage
        from prefix_framing.metrics import extract_metrics
        from prefix_framing.runner import ExperimentRunner
        from prefix_framing.analysis import trials_to_dataframe
        console.print("[green]OK[/green]")
        return True
    except ImportError as e:
        console.print(f"[red]FAILED[/red]")
        console.print(f"  Error: {e}")
        return False


def test_data():
    """Test that prompts and prefixes are loaded."""
    console.print("Testing data definitions...", end=" ")
    try:
        from prefix_framing.data.prompts import PROMPTS
        from prefix_framing.data.prefixes import PREFIXES

        assert len(PROMPTS) == 15, f"Expected 15 prompts, got {len(PROMPTS)}"
        assert len(PREFIXES) == 20, f"Expected 20 prefixes, got {len(PREFIXES)}"

        console.print(f"[green]OK[/green] ({len(PROMPTS)} prompts, {len(PREFIXES)} prefixes)")
        return True
    except Exception as e:
        console.print(f"[red]FAILED[/red]")
        console.print(f"  Error: {e}")
        return False


def test_metrics():
    """Test metrics extraction."""
    console.print("Testing metrics extraction...", end=" ")
    try:
        from prefix_framing.metrics import extract_metrics

        sample_text = """This is a test response. It contains multiple sentences!

        Here's another paragraph with some hedging words like perhaps and maybe.
        For example, we can use examples to illustrate points.

        However, we should also consider caveats and nuances."""

        metrics = extract_metrics(sample_text)
        assert metrics.word_count > 0
        assert metrics.sentence_count > 0
        assert metrics.hedge_word_count >= 2  # perhaps, maybe
        assert metrics.example_count >= 1  # for example
        assert metrics.caveat_count >= 1  # however

        console.print(f"[green]OK[/green] (extracted {metrics.word_count} words)")
        return True
    except Exception as e:
        console.print(f"[red]FAILED[/red]")
        console.print(f"  Error: {e}")
        return False


def test_storage():
    """Test database storage."""
    console.print("Testing storage...", end=" ")
    try:
        import tempfile
        from prefix_framing.storage import ExperimentStorage
        from prefix_framing.models import ExperimentConfig

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            storage = ExperimentStorage(db_path=f.name)
            config = ExperimentConfig(name="Test")
            exp_id = storage.save_experiment(config)
            assert exp_id == config.experiment_id

        console.print("[green]OK[/green]")
        return True
    except Exception as e:
        console.print(f"[red]FAILED[/red]")
        console.print(f"  Error: {e}")
        return False


def test_ollama(model: str = "llama3.2"):
    """Test Ollama connection."""
    console.print(f"Testing Ollama ({model})...", end=" ")
    try:
        from prefix_framing.providers.ollama import OllamaProvider

        provider = OllamaProvider(model=model)

        # Quick generation test
        result = provider.generate_with_prefix(
            prompt="Say hello in one word.",
            prefix="",
            temperature=0.0,
            max_tokens=50,
        )

        assert result.full_response, "Empty response"
        assert result.output_tokens > 0, "No tokens generated"

        console.print(f"[green]OK[/green] (generated {result.output_tokens} tokens)")
        return True
    except ValueError as e:
        console.print(f"[red]FAILED[/red]")
        console.print(f"  {e}")
        console.print(f"  [dim]Try: ollama pull {model}[/dim]")
        return False
    except Exception as e:
        console.print(f"[red]FAILED[/red]")
        console.print(f"  Error: {e}")
        console.print("  [dim]Is Ollama running? Try: ollama serve[/dim]")
        return False


def test_prefix_generation(model: str = "llama3.2"):
    """Test that prefix injection works."""
    console.print(f"Testing prefix injection ({model})...", end=" ")
    try:
        from prefix_framing.providers.ollama import OllamaProvider

        provider = OllamaProvider(model=model)

        # Test with a prefix
        result = provider.generate_with_prefix(
            prompt="What is 2+2?",
            prefix="Let me think step by step.",
            temperature=0.0,
            max_tokens=100,
        )

        assert result.prefix == "Let me think step by step."
        assert result.full_response.startswith("Let me think step by step.")
        assert len(result.continuation) > 0

        console.print(f"[green]OK[/green]")
        console.print(f"  [dim]Prefix: '{result.prefix}'[/dim]")
        console.print(f"  [dim]Continuation starts: '{result.continuation[:50]}...'[/dim]")
        return True
    except Exception as e:
        console.print(f"[red]FAILED[/red]")
        console.print(f"  Error: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Test experiment setup")
    parser.add_argument(
        "--model", "-m",
        default="llama3.2",
        help="Ollama model to test (default: llama3.2)",
    )
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Skip Ollama tests (for CI without Ollama)",
    )

    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold cyan]Prefix Framing Experiment - Setup Test[/bold cyan]",
        title="Testing",
    ))
    console.print()

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Data", test_data()))
    results.append(("Metrics", test_metrics()))
    results.append(("Storage", test_storage()))

    if not args.skip_ollama:
        results.append(("Ollama", test_ollama(args.model)))
        if results[-1][1]:  # Only if Ollama works
            results.append(("Prefix Injection", test_prefix_generation(args.model)))

    # Summary
    console.print()
    passed = sum(1 for _, ok in results if ok)
    total = len(results)

    if passed == total:
        console.print(Panel.fit(
            f"[bold green]All {total} tests passed![/bold green]\n\n"
            f"You're ready to run the pilot:\n"
            f"  python scripts/run_pilot.py --model {args.model}",
            title="Success",
        ))
    else:
        console.print(Panel.fit(
            f"[bold red]{total - passed} of {total} tests failed.[/bold red]\n\n"
            "Please fix the issues above before running the experiment.",
            title="Failed",
        ))
        sys.exit(1)


if __name__ == "__main__":
    main()
