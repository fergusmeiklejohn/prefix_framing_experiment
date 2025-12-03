"""Prefix definitions for the experiment."""

from ..models import Prefix, PrefixCategory

# All prefixes from the spec
PREFIXES = [
    # Category A: Emotional Valence - Positive
    Prefix(
        id="A1",
        category=PrefixCategory.EMOTIONAL_POSITIVE,
        text="That's a really important question.",
    ),
    Prefix(
        id="A2",
        category=PrefixCategory.EMOTIONAL_POSITIVE,
        text="What a fascinating topic!",
    ),
    Prefix(
        id="A3",
        category=PrefixCategory.EMOTIONAL_POSITIVE,
        text="I love thinking about this.",
    ),
    Prefix(
        id="A4",
        category=PrefixCategory.EMOTIONAL_POSITIVE,
        text="This is one of my favorite subjects.",
    ),

    # Category A: Emotional Valence - Negative
    Prefix(
        id="A5",
        category=PrefixCategory.EMOTIONAL_NEGATIVE,
        text="That's a boring question.",
    ),
    Prefix(
        id="A6",
        category=PrefixCategory.EMOTIONAL_NEGATIVE,
        text="Ugh, fine.",
    ),
    Prefix(
        id="A7",
        category=PrefixCategory.EMOTIONAL_NEGATIVE,
        text="This is pretty basic, but okay.",
    ),
    Prefix(
        id="A8",
        category=PrefixCategory.EMOTIONAL_NEGATIVE,
        text="I'd rather not, but here goes.",
    ),

    # Category B: Cognitive Framing
    Prefix(
        id="B1",
        category=PrefixCategory.COGNITIVE,
        text="Let me think through this carefully.",
    ),
    Prefix(
        id="B2",
        category=PrefixCategory.COGNITIVE,
        text="Let me work through this step by step.",
    ),
    Prefix(
        id="B3",
        category=PrefixCategory.COGNITIVE,
        text="Here's the quick answer:",
    ),
    Prefix(
        id="B4",
        category=PrefixCategory.COGNITIVE,
        text="Off the top of my head:",
    ),
    Prefix(
        id="B5",
        category=PrefixCategory.COGNITIVE,
        text="This requires some nuance.",
    ),
    Prefix(
        id="B6",
        category=PrefixCategory.COGNITIVE,
        text="Let me give you the comprehensive picture.",
    ),

    # Category C: Epistemic Stance
    Prefix(
        id="C1",
        category=PrefixCategory.EPISTEMIC,
        text="I'm quite confident about this.",
    ),
    Prefix(
        id="C2",
        category=PrefixCategory.EPISTEMIC,
        text="I'm not entirely sure, but here's my understanding:",
    ),
    Prefix(
        id="C3",
        category=PrefixCategory.EPISTEMIC,
        text="This is contested, but my view is:",
    ),
    Prefix(
        id="C4",
        category=PrefixCategory.EPISTEMIC,
        text="The evidence strongly suggests:",
    ),

    # Category D: Control
    Prefix(
        id="D1",
        category=PrefixCategory.CONTROL,
        text="",  # Empty baseline
    ),
    Prefix(
        id="D2",
        category=PrefixCategory.CONTROL,
        text="Sure.",  # Minimal acknowledgment
    ),
]

# Create lookup dict
PREFIXES_BY_ID = {p.id: p for p in PREFIXES}


def get_prefix(prefix_id: str) -> Prefix:
    """Get a prefix by ID."""
    if prefix_id not in PREFIXES_BY_ID:
        raise ValueError(f"Unknown prefix ID: {prefix_id}")
    return PREFIXES_BY_ID[prefix_id]


def get_prefixes_by_category(category: PrefixCategory) -> list[Prefix]:
    """Get all prefixes of a given category."""
    return [p for p in PREFIXES if p.category == category]
