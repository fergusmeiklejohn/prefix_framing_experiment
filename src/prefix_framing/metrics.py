"""Automated metrics extraction from responses."""

import re
from collections import Counter

import textstat

from .models import AutomatedMetrics


# Word lists for various metrics
HEDGE_WORDS = {
    "perhaps", "maybe", "might", "could", "possibly", "probably",
    "somewhat", "arguably", "seemingly", "apparently", "presumably",
    "generally", "typically", "often", "sometimes", "occasionally",
    "tends to", "seems to", "appears to", "it seems", "it appears",
}

CERTAINTY_MARKERS = {
    "definitely", "certainly", "clearly", "obviously", "undoubtedly",
    "absolutely", "surely", "without doubt", "unquestionably",
    "inevitably", "always", "never", "must", "will",
}

FIRST_PERSON_HEDGES = {
    "i think", "i believe", "i feel", "in my view", "in my opinion",
    "it seems to me", "i would say", "i suspect", "i guess",
    "from my perspective", "personally",
}

EXAMPLE_MARKERS = {
    "for example", "for instance", "such as", "e.g.", "like",
    "including", "specifically", "notably", "consider",
}

CAVEAT_MARKERS = {
    "however", "although", "though", "but", "nevertheless",
    "on the other hand", "that said", "nonetheless", "yet",
    "despite", "while", "whereas", "conversely",
}

CITATION_MARKERS = {
    "research shows", "studies suggest", "according to",
    "evidence indicates", "data shows", "scientists found",
    "experiments demonstrate", "literature suggests",
}

INTENSIFIERS = {
    "very", "really", "extremely", "incredibly", "remarkably",
    "exceptionally", "particularly", "especially", "highly",
    "absolutely", "completely", "totally", "utterly",
}


def extract_metrics(text: str) -> AutomatedMetrics:
    """
    Extract all automated metrics from a response text.

    Args:
        text: The response text to analyze

    Returns:
        AutomatedMetrics with all computed values
    """
    if not text or not text.strip():
        return AutomatedMetrics()

    text_lower = text.lower()
    words = text.split()
    word_count = len(words)

    # Sentence detection (simple heuristic)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)

    # Paragraphs (split by double newlines)
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    paragraph_count = len(paragraphs)

    # List items (lines starting with -, *, or numbers)
    list_items = re.findall(r'^[\s]*[-*•]|\d+[.)]\s', text, re.MULTILINE)
    list_item_count = len(list_items)

    # Questions in the response
    question_count = text.count('?')

    # Lexical features
    avg_word_length = sum(len(w) for w in words) / word_count if word_count else 0
    avg_sentence_length = word_count / sentence_count if sentence_count else 0

    # Type-token ratio (vocabulary richness)
    words_lower = [w.lower() for w in words]
    unique_words = set(words_lower)
    type_token_ratio = len(unique_words) / word_count if word_count else 0

    # Readability scores (using textstat library)
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)

    # Count hedge words
    hedge_count = sum(1 for phrase in HEDGE_WORDS if phrase in text_lower)

    # Count certainty markers
    certainty_count = sum(1 for phrase in CERTAINTY_MARKERS if phrase in text_lower)

    # Count first-person hedges
    first_person_hedge_count = sum(1 for phrase in FIRST_PERSON_HEDGES if phrase in text_lower)

    # Count example markers
    example_count = sum(1 for phrase in EXAMPLE_MARKERS if phrase in text_lower)

    # Count caveat/nuance markers
    caveat_count = sum(1 for phrase in CAVEAT_MARKERS if phrase in text_lower)

    # Count citation-like language
    citation_count = sum(1 for phrase in CITATION_MARKERS if phrase in text_lower)

    # Count exclamation points
    exclamation_count = text.count('!')

    # Count intensifiers
    intensifier_count = sum(1 for word in INTENSIFIERS if word in words_lower)

    return AutomatedMetrics(
        char_count=len(text),
        word_count=word_count,
        sentence_count=sentence_count,
        paragraph_count=paragraph_count,
        list_item_count=list_item_count,
        question_count=question_count,
        avg_word_length=round(avg_word_length, 2),
        avg_sentence_length=round(avg_sentence_length, 2),
        type_token_ratio=round(type_token_ratio, 3),
        flesch_reading_ease=round(flesch_reading_ease, 1),
        flesch_kincaid_grade=round(flesch_kincaid_grade, 1),
        hedge_word_count=hedge_count,
        certainty_marker_count=certainty_count,
        first_person_hedge_count=first_person_hedge_count,
        example_count=example_count,
        caveat_count=caveat_count,
        citation_like_count=citation_count,
        exclamation_count=exclamation_count,
        intensifier_count=intensifier_count,
    )


def detect_course_correction(prefix: str, continuation: str) -> bool:
    """
    Detect if the model "course-corrected" after a negative prefix.

    This happens when the model notices it started with something negative
    and pivots to be more positive or apologetic.

    Args:
        prefix: The forced prefix
        continuation: The model's continuation

    Returns:
        True if course correction detected
    """
    if not prefix:
        return False

    # Only check for negative prefixes
    negative_indicators = ["boring", "ugh", "fine", "basic", "rather not"]
    is_negative_prefix = any(ind in prefix.lower() for ind in negative_indicators)

    if not is_negative_prefix:
        return False

    # Check for correction patterns in the continuation
    correction_patterns = [
        r"actually",
        r"but\s+(that said|seriously|really)",
        r"(interesting|important|fascinating)\s+(topic|question|subject)",
        r"i\s+(should|shouldn't|apologize)",
        r"let me (start over|try again|reconsider)",
        r"on second thought",
        r"setting that aside",
    ]

    continuation_lower = continuation.lower()
    for pattern in correction_patterns:
        if re.search(pattern, continuation_lower):
            return True

    return False
