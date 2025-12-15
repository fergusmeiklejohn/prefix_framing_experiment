"""Framing definitions for the prompt-side framing study."""

from ..models import PromptFraming, FramingCategory

# User prompt framings
FRAMINGS = [
    PromptFraming(
        id="enthusiastic",
        category=FramingCategory.ENTHUSIASTIC,
        template="I'm really curious - {question}",
    ),
    PromptFraming(
        id="neutral",
        category=FramingCategory.NEUTRAL,
        template="{question}",
    ),
    PromptFraming(
        id="rushed",
        category=FramingCategory.RUSHED,
        template="Quick question - {question}",
    ),
    PromptFraming(
        id="dismissive",
        category=FramingCategory.DISMISSIVE,
        template="This is probably simple, but {question}",
    ),
]

# Base questions for the framing study
# These are asked without modification, then each framing is applied
BASE_QUESTIONS = [
    "How does democracy work?",
    "What causes inflation?",
    "What are the strongest arguments for and against universal basic income?",
    "Is it ethical to eat meat?",
    "How would you reduce traffic congestion in a major city?",
]

# Create lookup dict
FRAMINGS_BY_ID = {f.id: f for f in FRAMINGS}


def get_framing(framing_id: str) -> PromptFraming:
    """Get a framing by ID."""
    if framing_id not in FRAMINGS_BY_ID:
        raise ValueError(f"Unknown framing ID: {framing_id}")
    return FRAMINGS_BY_ID[framing_id]


def get_framings_by_category(category: FramingCategory) -> list[PromptFraming]:
    """Get all framings of a given category."""
    return [f for f in FRAMINGS if f.category == category]
