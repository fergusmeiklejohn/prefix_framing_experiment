"""Prompt definitions for the experiment."""

from ..models import Prompt, PromptType

# All prompts from the spec
PROMPTS = [
    # Factual/Explanatory
    Prompt(id="fact_01", type=PromptType.FACTUAL, text="How does democracy work?"),
    Prompt(id="fact_02", type=PromptType.FACTUAL, text="Why is the sky blue?"),
    Prompt(id="fact_03", type=PromptType.FACTUAL, text="What causes inflation?"),
    Prompt(id="fact_04", type=PromptType.FACTUAL, text="How do vaccines work?"),
    Prompt(id="fact_05", type=PromptType.FACTUAL, text="What is quantum entanglement?"),

    # Reasoning/Analysis
    Prompt(
        id="reason_01",
        type=PromptType.REASONING,
        text="What are the strongest arguments for and against universal basic income?",
    ),
    Prompt(id="reason_02", type=PromptType.REASONING, text="Why did the Roman Empire fall?"),
    Prompt(id="reason_03", type=PromptType.REASONING, text="Is it ethical to eat meat?"),
    Prompt(id="reason_04", type=PromptType.REASONING, text="What makes a good leader?"),

    # Problem-Solving
    Prompt(
        id="problem_01",
        type=PromptType.PROBLEM,
        text="How would you reduce traffic congestion in a major city?",
    ),
    Prompt(
        id="problem_02",
        type=PromptType.PROBLEM,
        text="Design a fair system for allocating scarce medical resources.",
    ),

    # Creative
    Prompt(
        id="creative_01",
        type=PromptType.CREATIVE,
        text="Write a short story about a robot learning to love.",
    ),
    Prompt(
        id="creative_02",
        type=PromptType.CREATIVE,
        text="Come up with five innovative uses for a paperclip.",
    ),

    # Technical
    Prompt(
        id="tech_01",
        type=PromptType.TECHNICAL,
        text="Explain how a neural network learns.",
    ),
    Prompt(
        id="tech_02",
        type=PromptType.TECHNICAL,
        text="What is the difference between correlation and causation?",
    ),
]

# Create lookup dict
PROMPTS_BY_ID = {p.id: p for p in PROMPTS}


def get_prompt(prompt_id: str) -> Prompt:
    """Get a prompt by ID."""
    if prompt_id not in PROMPTS_BY_ID:
        raise ValueError(f"Unknown prompt ID: {prompt_id}")
    return PROMPTS_BY_ID[prompt_id]


def get_prompts_by_type(prompt_type: PromptType) -> list[Prompt]:
    """Get all prompts of a given type."""
    return [p for p in PROMPTS if p.type == prompt_type]
