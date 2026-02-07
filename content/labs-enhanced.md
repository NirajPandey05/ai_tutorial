Labs — Enhanced Companion

This companion highlights core labs across the repository (agents, embeddings, fine-tuning, RAG) and supplements them with diagrams, videos, and practical tips.

1) Retrieval & Embeddings
- Core idea: map text to vector embeddings and use similarity search to retrieve relevant context.
- Helpful diagram: vector search intuition — https://ml6team.github.io/images/ann_vector_search.png
- Video: "Vector Databases and Embeddings explained" — Harrison Chase (Pinecone): https://www.youtube.com/watch?v=ZJ7m7XJzssw

Suggested lab improvements:
- Add an interactive notebook that visualizes embeddings in 2D (t-SNE/UMAP) with sample queries.

2) Fine-tuning & Dataset prep
- Concepts: dataset formats, batching, validation split, monitoring metrics, and stopping criteria.
- Video walkthrough: "Fine-tuning LLMs" (conceptual overview) — Hugging Face or DeepLearning.AI playlists.

3) RAG (Retrieval-Augmented Generation)
- Diagram: RAG pipeline (retriever + reader/generator) — https://docs.pinecone.io/docs/images/rag.png
- Video: "How RAG works" — LangChain or Retrieval-Augmented Generation tutorials on YouTube.

4) Agents & Tooling labs
- Visual: agent loop (observe -> think -> act -> tool) diagram: https://miro.medium.com/max/1400/1*XcXfQ6UX2f8wG8Q-HH0Jrg.png
- Add step-by-step debug notebooks showing tool calls, function schemas, and mock tool execution traces.

General suggestions for all labs
- Include at least one diagram per lab (inline image) and 1–2 short video links.
- Provide a "quick start" notebook per lab that runs a small end-to-end example and produces visualizations (e.g., embedding scatter plot, retrieval results table, confusion/metric charts).

Next steps
- Confirm if these enhanced companion files should be merged back into each original markdown file (in-place edits) or kept as separate companion resources.
- On confirmation, proceed to either edit files in-place to add images/video links or create per-lab notebooks and media assets.
