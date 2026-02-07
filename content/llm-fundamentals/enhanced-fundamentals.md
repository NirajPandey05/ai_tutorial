LLM Fundamentals — Enhanced Overview

This enhanced companion synthesizes the key concepts from the llm-fundamentals folder (Intro, Parameters, Prompt Engineering, Tokens) and provides visual resources and guided videos to aid learning.

1) What are LLMs — Quick summary
- Large language models (LLMs) are neural networks trained on large text corpora to predict tokens; they generalize across many language tasks.
- Key concept: forecast the next token using learned representations.

Visual resources:
- Illustrated Transformer (visual walkthrough of Transformer architectures): https://jalammar.github.io/illustrated-transformer/
- Attention diagram image: https://jalammar.github.io/images/transformer_self_attention.png

Video resources:
- "Attention Is All You Need" summarized (paper walkthrough) — Yannic Kilcher: https://www.youtube.com/watch?v=U0s0f995w14
- "The Illustrated Transformer" (presentation) — Jay Alammar: https://www.youtube.com/watch?v=4Bdc55j80l8

2) How they work — Short explanation
- Input tokens are converted to embeddings, processed by multi-head self-attention + feed-forward layers, and decoded into output tokens.
- Important building blocks: tokenization, positional encoding, attention, residual connections, layer normalization.

Suggested image:
- Tokenization and embedding pipeline diagram: https://miro.medium.com/max/1400/1*6xq3XzG0b7c0V1IV0Qqg6g.png

3) Parameters & temperature, stop sequences, token limits
- Temperature controls randomness (0 = deterministic, higher => more diverse outputs).
- Stop sequences end generation early; specify exact strings.
- Token limits bound context window and output length; be mindful of tokenization differences between models.

Interactive tools:
- Token counters and playgrounds (local or web) are invaluable to experiment with temperature and stop sequences.

4) Prompt engineering — Practical tips
- Start with a concise system or instruction, provide examples (few-shot), and use constraints (format specification) for reliable structured output.
- Use step-by-step decomposition for complex tasks and chain-of-thought when you need reasoning.

Prompt patterns reference:
- Zero-/few-shot and chain-of-thought patterns: https://www.promptingguide.ai/

5) Tokens & context management
- Use chunking strategies to keep important context in the active window.
- When context overflows, use retrieval-augmented generation (RAG) or summarization to compress context.

6) Quick labs to try (from repository)
- First API call exercise: observe prompts vs completions and experiment with temperature/stop sequences.
- Tokenization lab: measure token counts for long inputs and see how they map to model limits.
- Prompt playground: convert instructions to structured output using JSON schemas and validate.

7) Additional readings and images
- "Attention is All You Need" paper: https://arxiv.org/abs/1706.03762
- Jay Alammar's Transformer explainer: https://jalammar.github.io/illustrated-transformer/

If desired, these summaries can be merged into the original files or expanded into slide decks with images embedded locally.
