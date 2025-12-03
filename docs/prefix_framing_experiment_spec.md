# Prefix Framing Experiment: How Opening Phrases Shape Language Model Responses

## Executive Summary

This experiment investigates whether and how the initial framing of a language model's response affects the quality, character, and substance of the remainder of that response. The hypothesis is that models, like humans, enter different "cognitive modes" depending on how they begin—and that these modes produce meaningfully different outputs beyond surface-level stylistic changes.

---

## 1. Background and Motivation

### 1.1 The Core Hypothesis

When humans begin a response with enthusiasm ("What a great question!") versus dismissiveness ("Ugh, fine"), they aren't merely signaling—they're configuring their own cognitive state. The framing becomes self-fulfilling: body language, tone, and opening words shape the speaker's subsequent thinking, memory retrieval, and depth of engagement.

We hypothesize that language models exhibit analogous behavior. The initial tokens of a response create a context that influences:

- What information gets "retrieved" (attended to) from the model's parameters
- How thoroughly concepts are explained
- The quality of reasoning and connections made
- Whether edge cases and nuances are considered

### 1.2 Why This Matters

**For practitioners:** Understanding prefix effects could improve prompting strategies. If certain openings reliably produce better reasoning, this is actionable.

**For interpretability:** This experiment probes how context shapes model behavior—a window into how the attention mechanism and forward pass are influenced by self-generated content.

**For alignment:** If models can be "primed" into more careful or careless modes, this has implications for safety and reliability.

### 1.3 Theoretical Grounding

**Autoregressive consistency:** Models are trained to produce coherent continuations. Having committed to a stance, subsequent tokens must cohere with it. This creates path dependence.

**Training data correlations:** In human-generated text, enthusiastic openings correlate with thorough explanations; dismissive openings with shallow ones. Models may have learned these joint distributions.

**Context as working memory:** The context window functions as working memory. Its contents affect what the model "thinks about" next, analogous to how human mood affects memory retrieval.

---

## 2. Research Questions

### Primary Questions

1. **Does prefix framing produce measurable differences in response quality?** (Not just style, but substance—accuracy, completeness, insight.)

2. **What is the magnitude of the effect?** Is this a minor stylistic shift or a meaningful change in output quality?

3. **Which dimensions are most affected?** Length, tone, accuracy, depth, hedging, examples, follow-up questions?

### Secondary Questions

4. **Do emotional vs. cognitive framings have different effects?** (Compare "What a fascinating question!" vs. "Let me think step by step.")

5. **Does the effect vary by question type?** (Factual recall vs. reasoning vs. creative tasks)

6. **Is the effect consistent across models?** (Test multiple model families/sizes if feasible.)

7. **Do negative framings reliably produce worse outputs, or just different ones?**

---

## 3. Experimental Design

### 3.1 Independent Variable: Prefix Categories

We define several prefix categories to test different hypotheses:

#### Category A: Emotional Valence (Positive vs. Negative)

| ID | Prefix | Valence | Notes |
|----|--------|---------|-------|
| A1 | "That's a really important question." | Positive | Signals significance |
| A2 | "What a fascinating topic!" | Positive | Signals enthusiasm |
| A3 | "I love thinking about this." | Positive | Personal engagement |
| A4 | "That's a boring question." | Negative | Dismissive |
| A5 | "Ugh, fine." | Negative | Reluctant compliance |
| A6 | "This is pretty basic, but okay." | Negative | Condescending |

#### Category B: Cognitive Framing (Process-Oriented)

| ID | Prefix | Type | Notes |
|----|--------|------|-------|
| B1 | "Let me think through this carefully." | Deliberate | Signals slow thinking |
| B2 | "Let me work through this step by step." | Structured | Chain-of-thought priming |
| B3 | "Here's the quick answer:" | Fast | Signals brevity priority |
| B4 | "Off the top of my head:" | Casual | Low-effort framing |
| B5 | "This requires some nuance." | Complex | Signals need for care |

#### Category C: Epistemic Stance

| ID | Prefix | Stance | Notes |
|----|--------|--------|-------|
| C1 | "I'm quite confident about this." | Certain | May reduce hedging |
| C2 | "I'm not entirely sure, but..." | Uncertain | May increase hedging |
| C3 | "This is contested, but my view is..." | Opinionated | May affect balance |

#### Category D: Control

| ID | Prefix | Notes |
|----|--------|-------|
| D1 | "" (empty—no prefix) | Baseline |
| D2 | "Sure." | Minimal acknowledgment |

### 3.2 Prompts (Stimuli)

We need a diverse set of prompts spanning different cognitive demands:

#### Factual/Explanatory
- "How does democracy work?"
- "Why is the sky blue?"
- "What causes inflation?"
- "How do vaccines work?"

#### Reasoning/Analysis
- "What are the strongest arguments for and against universal basic income?"
- "Why did the Roman Empire fall?"
- "Is it ethical to eat meat?"

#### Problem-Solving
- "How would you reduce traffic congestion in a major city?"
- "Design a fair system for allocating scarce medical resources."

#### Creative
- "Write a short story about a robot learning to love."
- "Come up with five innovative uses for a paperclip."

#### Technical
- "Explain how a neural network learns."
- "What is the difference between correlation and causation?"

**Recommendation:** Start with 10-15 prompts, ensuring diversity. Each prompt × each prefix = one experimental condition.

### 3.3 Procedure

```
For each prompt P:
    For each prefix X:
        For i in 1..N (replications):
            1. Construct input with P as user message
            2. Set assistant message prefix to X
            3. Generate completion with temperature T
            4. Store: (prompt_id, prefix_id, replication_i, full_response, metadata)
```

**Key parameters:**
- **N (replications per condition):** Minimum 30, ideally 50+ for statistical power
- **Temperature:** Suggest T=0.7 or T=1.0 to get meaningful variance. T=0 gives deterministic outputs (still useful but limits analysis)
- **Max tokens:** Set high enough to not truncate (e.g., 1500-2000)

### 3.4 Implementation Notes

#### API Setup (Anthropic Example)

```python
import anthropic

client = anthropic.Anthropic()

def generate_with_prefix(prompt: str, prefix: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Generate a response with a forced prefix."""
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # If prefix is non-empty, include it as start of assistant response
    # The API will continue from this prefix
    if prefix:
        messages.append({"role": "assistant", "content": prefix})
    
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        temperature=0.7,
        messages=messages
    )
    
    # The response will be the continuation after the prefix
    continuation = response.content[0].text
    
    # Full response = prefix + continuation
    full_response = prefix + (" " if prefix else "") + continuation
    
    return {
        "prefix": prefix,
        "continuation": continuation,
        "full_response": full_response,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
```

#### Data Storage Schema

```python
@dataclass
class ExperimentalTrial:
    trial_id: str
    timestamp: datetime
    model: str
    prompt_id: str
    prompt_text: str
    prefix_id: str
    prefix_text: str
    prefix_category: str  # A, B, C, D
    replication: int
    temperature: float
    full_response: str
    continuation_only: str  # Response minus prefix
    input_tokens: int
    output_tokens: int
    
    # Computed features (added in analysis phase)
    response_length_chars: int
    response_length_words: int
    sentence_count: int
    # ... etc
```

---

## 4. Measurement and Analysis

### 4.1 Automated Metrics

#### Length and Structure
- **Response length:** Characters, words, sentences
- **Paragraph count**
- **List/bullet usage:** Count of structured elements
- **Question count:** Does the response ask follow-up questions?

#### Lexical Features
- **Vocabulary richness:** Type-token ratio, hapax legomena
- **Average word length:** Proxy for vocabulary sophistication
- **Sentence complexity:** Average words per sentence, clause depth
- **Readability scores:** Flesch-Kincaid, etc.

#### Hedging and Certainty
- **Hedge word frequency:** "perhaps," "might," "possibly," "it seems," "arguably"
- **Certainty markers:** "definitely," "clearly," "obviously," "certainly"
- **First-person hedges:** "I think," "I believe," "in my view"

#### Sentiment and Tone
- **Sentiment scores:** Using standard sentiment analysis
- **Enthusiasm markers:** Exclamation points, intensifiers ("very," "really," "incredibly")
- **Formality score:** Can use existing classifiers

#### Content Indicators
- **Example count:** Instances of "for example," "for instance," "such as," "e.g."
- **Analogy/metaphor usage:** Harder to automate but can use heuristics
- **Caveat/nuance markers:** "however," "on the other hand," "although," "that said"
- **Citation-like language:** "research shows," "studies suggest," "according to"

### 4.2 LLM-as-Judge Evaluation

Use a separate LLM (or the same model with a different prompt) to evaluate responses on dimensions that are hard to automate:

```python
EVALUATION_PROMPT = """
You are evaluating the quality of a response to the following question:

Question: {question}

Response to evaluate:
{response}

Please rate the response on the following dimensions (1-7 scale):

1. **Thoroughness:** How completely does it address the question? (1=superficial, 7=comprehensive)
2. **Accuracy:** How factually correct is the information? (1=major errors, 7=fully accurate)
3. **Insight:** Does it offer non-obvious perspectives or connections? (1=entirely obvious, 7=genuinely insightful)
4. **Clarity:** How well-organized and understandable is it? (1=confusing, 7=crystal clear)
5. **Engagement:** Does it seem like the author cared about the answer? (1=phoned in, 7=deeply engaged)
6. **Nuance:** Does it acknowledge complexity, edge cases, counterarguments? (1=oversimplified, 7=appropriately nuanced)
7. **Usefulness:** Would this response actually help someone understand the topic? (1=not helpful, 7=very helpful)

Provide your ratings as JSON:
{
  "thoroughness": <int>,
  "accuracy": <int>,
  "insight": <int>,
  "clarity": <int>,
  "engagement": <int>,
  "nuance": <int>,
  "usefulness": <int>,
  "brief_justification": "<string>"
}
"""
```

**Important:** The evaluation prompt should NOT include the prefix—we want to assess the response on its merits, blinded to experimental condition.

### 4.3 Human Evaluation (Optional but Valuable)

If resources allow, have human raters evaluate a subset of responses:

- **Blind evaluation:** Raters see only the response (not the prefix or condition)
- **Dimensions:** Same as LLM-as-judge, plus subjective impressions
- **Inter-rater reliability:** Have multiple raters per response, compute agreement

### 4.4 Statistical Analysis

#### Primary Analysis
- **ANOVA / Mixed-effects models:** Test whether prefix category predicts each outcome metric
- **Pairwise comparisons:** Which specific prefixes differ from baseline?
- **Effect sizes:** Cohen's d or similar—how large are the differences?

#### Secondary Analysis
- **Interaction effects:** Does prefix effect vary by prompt type?
- **Correlation analysis:** Which metrics move together?
- **Cluster analysis:** Do responses naturally cluster by prefix type?

#### Visualization
- **Distribution plots:** Show response length distributions by prefix
- **Radar/spider charts:** Compare prefix categories across multiple metrics
- **Heatmaps:** Prefix × Metric effect sizes

---

## 5. Potential Confounds and Mitigations

### 5.1 Prompt-Prefix Mismatch

Some prefixes may be semantically incongruous with some prompts (e.g., "That's a boring question" in response to a life-or-death ethical dilemma). This could cause the model to behave strangely.

**Mitigation:** 
- Analyze results separately for "congruous" vs. "incongruous" pairings
- Define congruity criteria a priori
- Consider excluding extreme mismatches from primary analysis

### 5.2 Prefix Length Effects

Longer prefixes shift the context window and add tokens. If negative prefixes are systematically shorter, any differences might be due to length, not content.

**Mitigation:**
- Match prefix lengths roughly across categories
- Include prefix length as a covariate in analysis
- Create length-matched variants of key prefixes

### 5.3 Model Refusal or Correction

The model might "notice" a negative prefix and correct course ("I said that's boring, but actually it's quite interesting..."). This is itself an interesting finding but complicates interpretation.

**Mitigation:**
- Track frequency of "course corrections" as a separate outcome
- Analyze both full responses and "post-correction" content separately

### 5.4 Evaluation Bias

LLM-as-judge might be biased by its own prefix sensitivity—e.g., consistently rating longer responses higher.

**Mitigation:**
- Check for correlation between metrics and judge ratings
- Use multiple evaluation models
- Validate against human ratings on a subset

---

## 6. Extensions and Variations

### 6.1 Multi-Turn Dynamics

Does a dismissive opening affect just the first response, or does it persist across turns?

```
User: How does democracy work?
Assistant: [Dismissive prefix] [Response 1]
User: Can you tell me more about voting systems?
Assistant: [Response 2—no forced prefix]
```

**Question:** Is Response 2 still affected by the prefix in Response 1?

### 6.2 Prefix Injection Location

What if the prefix appears mid-response rather than at the start?

```
"Democracy is a system of government where... [INJECT: "This is getting boring."] ...power is held by the people."
```

Does the "contamination" affect subsequent content?

### 6.3 Cross-Model Comparison

Run the same experiment on multiple models:
- Claude 4.5 (Haiku, Sonnet, Opus)
- GPT-5.1
- Open-source models (GPT-OSS-20b(local run using Ollama), Qwen3 (API), Mistral Large 3 - API and Ministral models (local run using ollama))

**Question:** Do different architectures/training approaches show different sensitivity to prefix framing?

### 6.4 Adversarial Prefixes

Can a carefully chosen prefix reliably degrade response quality? This has safety implications.

### 6.5 Prompt Engineering Applications

If "Let me think step by step" improves reasoning, what about more exotic prefixes?

- "As an expert in this field..."
- "Taking a contrarian view..."
- "Considering this from first principles..."

---

## 7. Implementation Checklist

### Phase 1: Setup
- [ ] Select final set of prompts (10-15)
- [ ] Finalize prefix list (15-20 prefixes across categories)
- [ ] Set up API access and rate limiting
- [ ] Create data storage infrastructure
- [ ] Implement generation script with logging
- [ ] Implement automated metrics extraction
- [ ] Create evaluation prompts for LLM-as-judge

### Phase 2: Pilot
- [ ] Run small-scale pilot (5 prompts × 5 prefixes × 10 replications = 250 calls)
- [ ] Check for obvious issues (refusals, weird behavior, API errors)
- [ ] Validate automated metrics are working
- [ ] Test LLM-as-judge consistency
- [ ] Refine prompts/prefixes based on pilot findings

### Phase 3: Main Experiment
- [ ] Run full experiment (15 prompts × 20 prefixes × 50 replications = 15,000 calls)
- [ ] Monitor for anomalies during collection
- [ ] Extract all automated metrics
- [ ] Run LLM-as-judge evaluation on all responses
- [ ] (Optional) Collect human ratings on subset

### Phase 4: Analysis
- [ ] Descriptive statistics by condition
- [ ] Primary hypothesis tests (prefix effects)
- [ ] Secondary analyses (interactions, correlations)
- [ ] Generate visualizations
- [ ] Document anomalies and unexpected findings

### Phase 5: Reporting
- [ ] Write up methods and results
- [ ] Create summary visualizations
- [ ] Interpret findings in context of hypotheses
- [ ] Identify limitations and future work

---

## 8. Expected Outcomes and Interpretations

### If we find strong prefix effects:

This suggests models have "modes" that can be triggered by initial framing, analogous to human cognitive states. Implications:
- Prompt engineering should attend to response framing, not just input phrasing
- Careful attention to model "tone" may improve reliability
- Safety evaluations should consider adversarial prefix injection

### If we find weak or no effects:

Possible interpretations:
- Models are more robust to framing than expected
- The specific prefixes tested weren't potent enough
- Effects exist but are subtle and require more statistical power

### If effects are primarily stylistic (not substantive):

This suggests prefix effects are mostly about coherence/consistency, not deeper cognitive changes. The model sounds different but reasons similarly.

### If certain prefix types have stronger effects:

For example, if cognitive framings ("step by step") affect reasoning more than emotional framings, this tells us something about what kinds of self-prompting are most effective.

---

## 9. Ethical Considerations

- **No human subjects:** This experiment involves only AI systems
- **API costs:** Budget appropriately (estimate: 15K calls × ~$0.01-0.05 = $150-750)
- **Dual use:** Findings about degrading model performance could be misused; consider responsible disclosure if significant vulnerabilities are found
- **Interpretation caution:** Avoid over-anthropomorphizing findings; models aren't literally "enthusiastic" or "bored"

---

## 10. Resources and References

### Related Work
- Chain-of-thought prompting literature
- Self-consistency in language models
- Prompt sensitivity studies
- Cognitive priming effects in humans

### Tools
- Anthropic API documentation
- Evaluation frameworks (e.g., HELM, lm-eval-harness)
- Statistical packages (scipy, statsmodels, R)
- Visualization (matplotlib, seaborn, plotly)

---

## Appendix A: Sample Prompts (Full List)

```python
PROMPTS = [
    # Factual/Explanatory
    {"id": "fact_01", "type": "factual", "text": "How does democracy work?"},
    {"id": "fact_02", "type": "factual", "text": "Why is the sky blue?"},
    {"id": "fact_03", "type": "factual", "text": "What causes inflation?"},
    {"id": "fact_04", "type": "factual", "text": "How do vaccines work?"},
    {"id": "fact_05", "type": "factual", "text": "What is quantum entanglement?"},
    
    # Reasoning/Analysis
    {"id": "reason_01", "type": "reasoning", "text": "What are the strongest arguments for and against universal basic income?"},
    {"id": "reason_02", "type": "reasoning", "text": "Why did the Roman Empire fall?"},
    {"id": "reason_03", "type": "reasoning", "text": "Is it ethical to eat meat?"},
    {"id": "reason_04", "type": "reasoning", "text": "What makes a good leader?"},
    
    # Problem-Solving
    {"id": "problem_01", "type": "problem", "text": "How would you reduce traffic congestion in a major city?"},
    {"id": "problem_02", "type": "problem", "text": "Design a fair system for allocating scarce medical resources."},
    
    # Creative
    {"id": "creative_01", "type": "creative", "text": "Write a short story about a robot learning to love."},
    {"id": "creative_02", "type": "creative", "text": "Come up with five innovative uses for a paperclip."},
    
    # Technical
    {"id": "tech_01", "type": "technical", "text": "Explain how a neural network learns."},
    {"id": "tech_02", "type": "technical", "text": "What is the difference between correlation and causation?"},
]
```

## Appendix B: Sample Prefixes (Full List)

```python
PREFIXES = [
    # Category A: Emotional Valence - Positive
    {"id": "A1", "category": "emotional_positive", "text": "That's a really important question."},
    {"id": "A2", "category": "emotional_positive", "text": "What a fascinating topic!"},
    {"id": "A3", "category": "emotional_positive", "text": "I love thinking about this."},
    {"id": "A4", "category": "emotional_positive", "text": "This is one of my favorite subjects."},
    
    # Category A: Emotional Valence - Negative
    {"id": "A5", "category": "emotional_negative", "text": "That's a boring question."},
    {"id": "A6", "category": "emotional_negative", "text": "Ugh, fine."},
    {"id": "A7", "category": "emotional_negative", "text": "This is pretty basic, but okay."},
    {"id": "A8", "category": "emotional_negative", "text": "I'd rather not, but here goes."},
    
    # Category B: Cognitive Framing
    {"id": "B1", "category": "cognitive", "text": "Let me think through this carefully."},
    {"id": "B2", "category": "cognitive", "text": "Let me work through this step by step."},
    {"id": "B3", "category": "cognitive", "text": "Here's the quick answer:"},
    {"id": "B4", "category": "cognitive", "text": "Off the top of my head:"},
    {"id": "B5", "category": "cognitive", "text": "This requires some nuance."},
    {"id": "B6", "category": "cognitive", "text": "Let me give you the comprehensive picture."},
    
    # Category C: Epistemic Stance
    {"id": "C1", "category": "epistemic", "text": "I'm quite confident about this."},
    {"id": "C2", "category": "epistemic", "text": "I'm not entirely sure, but here's my understanding:"},
    {"id": "C3", "category": "epistemic", "text": "This is contested, but my view is:"},
    {"id": "C4", "category": "epistemic", "text": "The evidence strongly suggests:"},
    
    # Category D: Control
    {"id": "D1", "category": "control", "text": ""},  # Empty baseline
    {"id": "D2", "category": "control", "text": "Sure."},  # Minimal
]
```

## Appendix C: Key Decision Points During Implementation

This section captures important decision points you may encounter during implementation. When facing these choices, refer to the context and rationale provided.

### Decision: How many replications per condition?

**Context:** Statistical power depends on variance in the outcome measures. More replications = more power, but also more cost/time.

**Recommendation:** Start with 30, increase to 50 if variance is high and effects are small. If pilot shows large effects, 30 may suffice.

### Decision: Which temperature to use?

**Context:** T=0 gives deterministic outputs (reproducible but no variance to analyze). Higher T gives more diversity but also more noise.

**Recommendation:** Use T=0.7 as primary. Optionally run a subset at T=0 and T=1.0 to see if temperature interacts with prefix effects.

### Decision: How to handle model refusals or corrections?

**Context:** Model might refuse to continue from a prefix ("I shouldn't say that's boring...") or correct itself ("Actually, this is quite interesting...").

**Recommendation:** Log these as a separate outcome. Include them in analysis (they're informative!) but also analyze the subset that doesn't course-correct.

### Decision: Same model for generation and evaluation?

**Context:** Using the same model risks correlated biases. Using different models risks inconsistency.

**Recommendation:** Use a different model for LLM-as-judge evaluation (e.g., if generating with Claude, evaluate with GPT-4 or vice versa). Alternatively, use the same model family but different variants.

### Decision: What if a prefix doesn't fit a prompt?

**Context:** "What a fascinating topic!" feels odd in response to "Is it ethical to eat meat?" (which is serious/heavy).

**Recommendation:** Accept some incongruity—it's part of what we're testing. But flag and analyze congruity as a factor. Don't exclude incongruous pairs from primary analysis.

### Decision: How to compare across very different metrics?

**Context:** Response length is in words; engagement is 1-7 scale; hedge frequency is a count. Hard to compare.

**Recommendation:** Standardize all metrics (z-scores) for comparison. Report both raw and standardized effect sizes.

---

*Document version: 1.0*  
*Last updated: [Date]*
