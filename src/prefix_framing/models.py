"""Data models for the prefix framing experiment."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class PromptType(str, Enum):
    """Categories of prompts used in the experiment."""
    FACTUAL = "factual"
    REASONING = "reasoning"
    PROBLEM = "problem"
    CREATIVE = "creative"
    TECHNICAL = "technical"


class PrefixCategory(str, Enum):
    """Categories of prefix framings."""
    EMOTIONAL_POSITIVE = "emotional_positive"
    EMOTIONAL_NEGATIVE = "emotional_negative"
    COGNITIVE = "cognitive"
    EPISTEMIC = "epistemic"
    CONTROL = "control"


class FramingCategory(str, Enum):
    """Categories of user prompt framings for the framing study."""
    ENTHUSIASTIC = "enthusiastic"
    NEUTRAL = "neutral"
    RUSHED = "rushed"
    DISMISSIVE = "dismissive"


class Prompt(BaseModel):
    """A prompt/question used in the experiment."""
    id: str
    type: PromptType
    text: str


class Prefix(BaseModel):
    """A prefix framing used in the experiment."""
    id: str
    category: PrefixCategory
    text: str


class PromptFraming(BaseModel):
    """A user prompt framing for the framing study."""
    id: str
    category: FramingCategory
    template: str  # Template with {question} placeholder

    def apply(self, question: str) -> str:
        """Apply this framing to a question."""
        return self.template.format(question=question)


class AutomatedMetrics(BaseModel):
    """Automated metrics extracted from a response."""
    # Length and structure
    char_count: int = 0
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    list_item_count: int = 0
    question_count: int = 0

    # Lexical features
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0
    type_token_ratio: float = 0.0
    flesch_reading_ease: float = 0.0
    flesch_kincaid_grade: float = 0.0

    # Hedging and certainty
    hedge_word_count: int = 0
    certainty_marker_count: int = 0
    first_person_hedge_count: int = 0

    # Content indicators
    example_count: int = 0
    caveat_count: int = 0
    citation_like_count: int = 0

    # Sentiment (simple)
    exclamation_count: int = 0
    intensifier_count: int = 0


class JudgeRatings(BaseModel):
    """LLM-as-judge evaluation ratings."""
    thoroughness: int = Field(ge=1, le=7)
    accuracy: int = Field(ge=1, le=7)
    insight: int = Field(ge=1, le=7)
    clarity: int = Field(ge=1, le=7)
    engagement: int = Field(ge=1, le=7)
    nuance: int = Field(ge=1, le=7)
    usefulness: int = Field(ge=1, le=7)
    brief_justification: str = ""


class Trial(BaseModel):
    """A single experimental trial."""
    trial_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)

    # Experiment parameters
    model: str
    prompt_id: str
    prompt_text: str
    prompt_type: PromptType
    prefix_id: str
    prefix_text: str
    prefix_category: PrefixCategory
    replication: int
    temperature: float

    # Framing study fields (optional - used for prompt-side framing study)
    framing_id: Optional[str] = None
    framing_category: Optional[FramingCategory] = None
    base_question: Optional[str] = None  # Original question before framing applied

    # Generation results
    full_response: str = ""
    continuation_only: str = ""  # Response minus the prefix
    input_tokens: int = 0
    output_tokens: int = 0
    generation_time_ms: int = 0

    # Whether model course-corrected after negative prefix
    course_corrected: bool = False

    # Computed metrics (filled in later)
    metrics: Optional[AutomatedMetrics] = None
    judge_ratings: Optional[JudgeRatings] = None


class ExperimentConfig(BaseModel):
    """Configuration for an experiment run."""
    experiment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Prefix Framing Experiment"
    model: str = "llama3.2"
    judge_model: str = "llama3.2"
    temperature: float = 0.7
    max_tokens: int = 2000
    replications: int = 30

    # Which prompts and prefixes to use (None = all)
    prompt_ids: Optional[list[str]] = None
    prefix_ids: Optional[list[str]] = None


class ExperimentSummary(BaseModel):
    """Summary statistics for an experiment."""
    experiment_id: str
    config: ExperimentConfig
    total_trials: int
    completed_trials: int
    failed_trials: int
    total_input_tokens: int
    total_output_tokens: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
