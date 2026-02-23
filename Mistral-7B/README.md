# Mistral-7B Legal Document Summarizer

This project contains tools to summarize long legal documents (specifically Supreme Court judgments) using the `mistralai/Mistral-7B-Instruct-v0.2` model, and to evaluate the quality of those summaries against references using metrics like ROUGE and BERTScore.

## Files

1. **`long.py`**: Legal Document Summarizer v5.1
   - Uses `Mistral-7B-Instruct` to summarize documents into 8 specific structured headings (e.g., Court and Citation, Parties, Facts of the Case, etc.).
   - Optimized with `accelerate` and half-precision (FP16) for memory efficiency.
   - Saves output as structured text or JSON.

2. **`evaluate.py`**: Summary Evaluation Suite
   - Compares generated summaries against reference summaries.
   - Computes ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum, and BERTScore metrics.
   - Generates visually appealing PDF reports (`evaluation_results.pdf`) with charts and tables without requiring a browser.

3. **`pt.py`**: Model Converter
   - Converts the `Mistral-7B-Instruct-v0.2` model from the Hugging Face cache to a single `.pt` file (~14GB in fp16).
   - Useful for sharing, uploading to cloud storage, or bypassing redownloads.

## Installation

Install the required dependencies using the `requirements.txt` file (ideally inside a virtual environment):

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generating Summaries

To summarize a legal document (e.g., `input_judgment.txt`), run:

```bash
python long.py input_judgment.txt
```

**Options:**
- `--output`: Specify custom output filename.
- `--format`: Choose `structured` (default) or `json`.
- `--max-chars`: Max input characters before truncating (default: 12,000).
- `--max-tokens`: Max tokens for generation (default: 1,200).
- `--cpu`: Force CPU inference.
- `--no-fp16`: Use full FP32 precision.

### 2. Evaluating Summaries

To evaluate a generated summary against a reference, run:

```bash
python evaluate.py generated.txt reference.txt
```

This will output metrics to the console and produce a comprehensive evaluation report in PDF format.

**Options:**
- `--no-bertscore`: Disables BERTScore evaluation.
- `--batch`: Evaluates entire directories covering multiple pairs. Example: `python evaluate.py --batch summaries_dir/ references_dir/`


