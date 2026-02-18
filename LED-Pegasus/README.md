
# LED-Pegasus Legal Summarizer

This project implements a legal document summarizer using a hybrid approach with **LED (Longformer Encoder-Decoder)** and **Pegasus** models. It is designed to handle long legal texts and produce concise, accurate summaries.

## Features
- **Hybrid Summarization**: Combines the strengths of LED (for long contexts) and Pegasus (for abstractive quality).
- **Hallucination Detection**: Checks for common legal hallucinations (e.g., wrong jurisdiction, fabricated case numbers).
- **Consistency Checking**: Verifies that the summary is semantically consistent with the original text.
- **Smart Extraction**: Pre-selects key sentences to improve model focus.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Akshay01718/LED-Pegasus.git
    cd LED-Pegasus
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Models:**
    Since the model weights are large, they are not included in the repository. Please download them from the following Drive link:
    
    **https://drive.google.com/drive/folders/14D-n6_9yXihAbgS-miIoHfaqKtYj3KNQ?usp=drive_link**
    
    Download the following files and place them in the root directory of the project:
    - `legal_led_model.pt`
    - `legal_pegasus_model.pt`

## Usage

You can run the summarizer using the `summarizer.py` script.

### Basic Usage (using HuggingFace downloads)
If you have internet access and haven't downloaded the `.pt` files, the script will automatically download the models from Hugging Face:
```bash
python summarizer.py input.txt --model hybrid
```

### Using Local Models (Recommended)
To use the downloaded `.pt` files (faster and offline-capable):
```bash
python summarizer.py input.txt --model hybrid --led-model-path legal_led_model.pt --pegasus-model-path legal_pegasus_model.pt
```

### Arguments
- `input_file`: Path to the input text file containing the legal document.
- `--model`: Choose the model strategy: `led`, `pegasus`, or `hybrid` (default: `hybrid` recommended).
- `--led-model-path`: Path to the local `.pt` file for the LED model.
- `--pegasus-model-path`: Path to the local `.pt` file for the Pegasus model.
- `--output`: (Optional) Custom output file path.
- `--no-fallback`: (Optional) Disable the extractive fallback mechanism.

## Project Structure
- `summarizer.py`: Main script for inference.
- `save_models.py`: Utility script to download and save models as `.pt` files.
- `evaluate.py`: Script for evaluating summaries against references (requires `rouge-score`).
