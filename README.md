# Legal Document Summarizer
This repository contains a comprehensive set of AI architectures and text processing methodologies designed to summarize complex Indian Supreme Court judgments and other legal documents. 
The project employs four distinct summarization pipelines, each leveraging different combinations of neural networks, NLP heuristics, and graph-based extraction techniques to generate accurate, factual, and structured summaries.
## Features
- **Multi-Pipeline Architecture**: Includes Causal LLMs, Encoder-Decoder cascades, and Extractive-Abstractive graphs.
- **Legal-Specific Preprocessing**: Tailored text cleaning for OCR noise and heuristic truncation for optimal model ingestion.
- **Hallucination Detection**: Built-in quality assurance mechanisms (Sentence-BERT similarity, jurisdiction correction) to ensure factual accuracy.
- **Extensive Evaluation Suite**: Built-in ROUGE and BERTScore evaluation with automated PDF report generation and visual charting.
---
## Summarization Pipelines
### 1. Mistral-7B Structured Summarization (`Mistral-7B/`)
Leverages a 7-billion parameter Causal Language Model (`mistralai/Mistral-7B-Instruct-v0.2`) for highly structured summaries via zero/few-shot instruction prompting.
- Uses **Mixed Precision (FP16)** to reduce VRAM requirements.
- Preprocesses text using regex and heuristic truncation.
- Applies **Greedy Decoding** for deterministic, factual legal extraction.
- Parses output into structured dictionaries.
### 2. LED-Pegasus Hybrid Pipeline (`LED-Pegasus/`)
An **Encoder-Decoder (Seq2Seq)** cascade designed for documents exceeding 10,000 tokens.
- **Smart Legal Extraction**: A deterministic algorithm pre-filters and scores vital sentences using keyword heuristics and positional bias.
- **Longformer Encoder-Decoder (LED)**: Compresses the long document using global attention masking.
- **Pegasus Abstractor**: Refines the intermediate LED output into a coherent summary.
- **Jurisdiction Correction**: Automatically replaces hallucinated US legal terms with Indian equivalents.
### 3. LegalBERT + Flan-T5 Pipeline (`google-flan_t5/`)
An Extractive-Abstractive graph-based approach.
- **Extractive Phase**: Uses `nlpaueb/legal-bert-base-uncased` to generate dense sentence representations, then applies a PageRank algorithm (TextRank variant) to extract central sentences.
- **Abstractive Phase**: Feeds the extracted sentences into `google/flan-t5-base` using greedy decoding and dynamic quantization for CPU acceleration.
### 4. Template-Based Extractor (`baby.py`)
A non-neural, rule-based extraction system.
- Uses Regex lookaround assertions to pinpoint specific metadata and legal holdings.
- Filters out argumentative sentences to isolate purely established legal rules.
---
## Fine-Tuning Methodologies
The repository includes scripts to fine-tune the models on specific legal corpora (e.g., `percins/IN-ABS`):
- **Seq2Seq Fine-Tuning**: Trains LED, Pegasus, and Flan-T5 using HuggingFace `Seq2SeqTrainer` with gradient accumulation and cosine annealing.
- **BERT Extractive Fine-Tuning**: Converts standard BERT into a binary sentence classifier using Word Overlap thresholding.
---
## Evaluation & Quality Assurance
- **Metrics**: Computes ROUGE (R1, R2, RL, RLsum) and BERTScore.
- **Quality Checks**: Includes a `ConsistencyChecker` (Sentence-BERT cosine similarity) and a `HallucinationDetector` (Regex-based source vs. summary set differences).
- **Reporting**: Automatically generates non-interactive PDF reports (`ReportLab`) with embedded Matplotlib visualizations (Radar charts, Confusion Bubbles).
