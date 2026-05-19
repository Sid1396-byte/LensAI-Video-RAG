# LensAI: Multimodal Video RAG
A custom-built, local Multimodal Retrieval-Augmented Generation (RAG) system that transforms long-form video content into a searchable, queryable database using Gemini 3.1 Flash Lite and Whisper.

## Key Features
- **Multimodal Context:** Synthesizes audio transcripts and visual frames for full-video understanding.
- **Hybrid Search:** Combines Vector Embeddings (Cosine Similarity) with TF-IDF Keyword matching.
- **System Metrics:** Evaluated using DeepEval for Faithfulness, Precision, and Recall.
- **Smart Ingestion:** Fixed-interval (10s) frame sampling to balance token costs with visual context.