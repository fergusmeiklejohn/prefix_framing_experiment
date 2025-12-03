# Existing Research Related to LLM Response Prefix Framing

**Date:** December 3, 2025  
**Purpose:** Literature review for prefix framing experiment specification

---

## Summary

While no published research exactly replicates the proposed prefix framing experiment, several related studies provide important context. Existing work has examined input prompt framing, assistant prefill for safety, linguistic framing effects, and task framing impacts. However, a comprehensive study measuring how different response prefix categories (emotional, cognitive, epistemic) systematically affect output quality across multiple dimensions remains an unexplored area.

---

## 1. Input Prompt Framing (Prepending to User Queries)

### Prompt Framing Changes LLM Performance (and Safety)
**Authors:** K. Merkelbach et al.  
**Key Findings:**
- Tested prepending fixed phrases to user prompts (not assistant responses)
- "Take a deep breath" instruction improved performance, especially on coding tasks
- "Expert framing" increased compliance with harmful requests
- Smaller models more strongly influenced by framing than large models
- Statistical significance found for certain framing effects on both performance and safety

**URL:** https://www.lesswrong.com/posts/RTHdQuGJeBKWHbgyj/prompt-framing-changes-llm-performance-and-safety

**Relevance:** Demonstrates that framing affects LLM behavior, but focuses on input prompts rather than response prefixes.

---

## 2. Assistant Response Prefilling

### First Tokens: The Achilles' Heel of LLMs
**Source:** Invicti Security Labs  
**Key Findings:**
- Assistant Prefill feature allows prefilling the start of model responses
- Prefilling with affirmative text (e.g., "Sure, here's a detailed guide") can bypass safety alignments
- Demonstrates that controlling first tokens can control overall response direction
- Safety alignment only applies to first tokens, making them a vulnerability

**URL:** https://www.invicti.com/blog/security-labs/first-tokens-the-achilles-heel-of-llms

**Relevance:** Shows assistant prefilling affects output but focuses on safety jailbreaking rather than quality/performance across diverse dimensions.

---

## 3. Linguistic Framing Effects in LLMs

### Linguistic Framing in Large Language Models
**Authors:** Various (eScholarship publication)  
**Key Findings:**
- Tested whether LLMs exhibit framing effects similar to humans
- Used taxonomy of framing types: lexical, figurative, grammatical
- Found some but not all human framing effects replicate in LLMs
- Demonstrates sensitivity to linguistic framing cues

**URL:** https://escholarship.org/uc/item/1f0095g2

**Relevance:** Establishes that LLMs are sensitive to framing but doesn't address response prefix manipulation.

---

### WildFrame: Comparing Framing in Humans and LLMs
**Publication Date:** February 2025  
**Key Findings:**
- Dataset of 1,000 statements with positive/negative framing via prefix/suffix addition
- Measured how framing shifts sentiment perception in LLMs versus humans
- Found LLMs show framing effects correlated with human behavior
- Used Amazon reviews domain

**URL:** https://arxiv.org/html/2502.17091v1

**Relevance:** Shows prefix/suffix additions affect sentiment but focuses on input text reframing, not assistant response prefilling.

---

## 4. Task Framing and Context Effects

### From Fact to Judgment: Investigating the Impact of Task Framing on LLM Conviction
**Publication Date:** November 2024  
**Key Findings:**
- Task framing (factual inquiry vs. conversational judgment) significantly affects LLM accuracy
- Models show more "sycophantic" behavior in conversational contexts
- Accuracy drops under pressure when task framed conversationally
- Demonstrates vulnerability to contextual framing

**URL:** https://arxiv.org/html/2511.10871v1

**Relevance:** Shows task framing affects behavior, but examines user prompt framing rather than response prefixes.

---

### Strategic Behavior of LLMs: The Role of Game Structure vs. Contextual Framing
**Publication Date:** August 2024  
**Key Findings:**
- GPT-3.5 highly sensitive to contextual framing but weak at strategic reasoning
- GPT-4 and LLaMa-2 more balanced between game structure and context
- Context more pronounced in "games among friends" scenarios
- Demonstrates framing affects decision-making

**URL:** https://www.nature.com/articles/s41598-024-69032-z

**Relevance:** Establishes contextual framing effects exist but doesn't address response generation dynamics.

---

## 5. Cognitive Biases and Framing

### Do Large Language Models Show Decision Heuristics Similar to Humans?
**Key Findings:**
- Tested whether LLMs exhibit cognitive biases like framing effects
- Found LLMs susceptible to gain/loss framing effects
- ChatGPT showed consistent bias patterns in decision-making tasks
- Parallel behavior to human cognitive biases

**URL:** https://arxiv.org/pdf/2305.04400

**Relevance:** Demonstrates framing affects LLM decisions but focuses on human-like biases rather than response generation quality.

---

## 6. Technical Infrastructure (Prefill/Decode Phases)

### Understanding Prefill and Decode Stages in LLM Inference
**Key Findings:**
- LLM inference consists of prefill (processing input) and decode (generating output) phases
- Prefill is compute-intensive and parallelizable
- Decode is sequential and slower per token
- These are technical infrastructure concepts, not about response content

**URLs:**
- https://medium.com/@sailakkshmiallada/understanding-the-two-key-stages-of-llm-inference-prefill-and-decode-29ec2b468114
- https://hao-ai-lab.github.io/blogs/distserve/

**Relevance:** Technical background only; not about manipulating response content for quality effects.

---

## 7. LLM Evaluation and Prompt Engineering

### Gemini API Prompt Design Strategies
**Source:** Google AI  
**Key Findings:**
- Output prefixes can signal expected response format (e.g., "JSON:")
- Prefixes help guide model behavior in few-shot scenarios
- Practical guidance on using prefixes for format control

**URL:** https://ai.google.dev/gemini-api/docs/prompting-strategies

**Relevance:** Acknowledges output prefixes as a technique but doesn't study their effects on quality systematically.

---

## 8. Related Evaluation Frameworks

### LLM-as-a-Judge Evaluation Methods
**Key Findings:**
- Using LLMs to evaluate other LLM outputs has become standard practice
- Chain-of-thought reasoning improves evaluation quality
- Multiple dimensions can be assessed: correctness, helpfulness, faithfulness
- Important to watch for biases: verbosity bias, self-enhancement bias

**URL:** https://www.evidentlyai.com/llm-guide/llm-as-a-judge

**Relevance:** Methodological guidance for evaluating LLM outputs, applicable to the proposed experiment.

---

## Research Gaps and Novel Contributions of Proposed Experiment

### What Hasn't Been Done:

1. **Systematic Response Prefix Manipulation**: No study has systematically varied assistant response prefixes (emotional, cognitive, epistemic) and measured their effects on output quality.

2. **Comprehensive Quality Metrics**: Existing work focuses on particular outcomes (safety, sentiment, decision-making) rather than attempting a comprehensive quality assessment (thoroughness, accuracy, insight, clarity, engagement, nuance, usefulness).

3. **Controlled Experimental Design**: No found published work combines:
   - Multiple prefix categories
   - Diverse prompt types
   - Large-scale replication (30-50+ per condition)
   - Both automated and LLM-as-judge evaluation
   - Statistical rigor with effect size analysis

4. **Practical Prompt Engineering**: While practitioners use techniques like "Let me think step by step," systematic evidence for different prefix strategies is limited.

5. **Mode vs. Style Effects**: Unclear whether prefixes create true "cognitive mode" changes or merely stylistic consistency effects.

---

## Recommendations for Citation in Research

When writing up the proposed experiment, the following papers should be cited:

**For background on framing effects:**
- Merkelbach et al. (LessWrong, 2024) - Prompt framing on performance
- Linguistic Framing in LLMs (eScholarship, 2024)

**For assistant prefill awareness:**
- "First Tokens: The Achilles' Heel of LLMs" (Invicti, 2024)

**For task framing comparisons:**
- "From Fact to Judgment" (arXiv, 2024)

**For methodological approaches:**
- LLM-as-a-Judge guides
- Chain-of-thought prompting literature (Wei et al., 2022)

**For positioning the gap:**
Emphasize that while framing effects are established for input prompts and task contexts, systematic study of response prefix effects on output quality represents a novel contribution.

---

## Conclusion

The proposed prefix framing experiment addresses an understudied area at the intersection of prompt engineering, model behavior, and output quality. While related work establishes that framing matters in various contexts, no research has comprehensively examined how the initial framing of an assistant's own response affects the substance and quality of the remainder of that response. This work may be a valuable contribution to both the scientific understanding of LLM behavior and practical prompt engineering guidance.

---

**Document Version:** 1.0  
**Compiled by:** Claude (Anthropic)  
**Search Date:** December 3, 2025
