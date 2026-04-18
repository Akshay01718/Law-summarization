#!/usr/bin/env python3
"""
Convert Mistral-7B-Instruct-v0.2 to .pt format
This creates a single .pt checkpoint file that can be uploaded to Git LFS or cloud storage
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def convert_mistral_to_pt():
    """
    Convert Mistral-7B-Instruct from HuggingFace cache to single .pt file.
    Creates: mistral_7b_instruct.pt (~14 GB in fp16)
    """
    
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    output_file = "mistral_7b_instruct.pt"
    
    logger.info("=" * 70)
    logger.info("Converting Mistral-7B-Instruct to .pt format")
    logger.info("=" * 70)
    
    # Step 1: Load the model from HuggingFace cache (already downloaded)
    logger.info(f"Loading model: {model_name}")
    logger.info("(This uses your existing HuggingFace cache, no new download)")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,  # Half precision = ~14 GB file
        device_map="cpu",     # Keep on CPU during conversion
        low_cpu_mem_usage=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    logger.info("✓ Model loaded from cache")
    
    # Step 2: Create checkpoint dictionary
    logger.info("Preparing checkpoint...")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': model.config.to_dict(),
        'model_name': model_name,
        'dtype': 'float16',
    }
    
    # Step 3: Save to .pt file
    logger.info(f"Saving to {output_file}...")
    torch.save(checkpoint, output_file)
    
    # Report file size
    file_size_gb = Path(output_file).stat().st_size / (1024**3)
    
    logger.info("=" * 70)
    logger.info(f"✅ SUCCESS")
    logger.info(f"Output: {output_file}")
    logger.info(f"Size  : {file_size_gb:.2f} GB")
    logger.info("=" * 70)
    logger.info("")
    logger.info("You can now upload this .pt file to:")
    logger.info("  • Git LFS (if repo supports large files)")
    logger.info("  • Google Drive / Dropbox / OneDrive")
    logger.info("  • Hugging Face Hub as a private model")
    logger.info("  • AWS S3 / Azure Blob / GCP Storage")
    logger.info("")
    logger.info("Your original HuggingFace cache can be deleted to save space:")
    logger.info("  rm -rf ~/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2")
    logger.info("=" * 70)


if __name__ == "__main__":
    convert_mistral_to_pt()