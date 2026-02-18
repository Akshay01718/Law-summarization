
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

def save_model_as_pt(model_name, output_filename):
    print(f"Loading {model_name}...")
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Save state dict
        print(f"Saving state dictionary to {output_filename}...")
        torch.save(model.state_dict(), output_filename)
        
        # Also save the config and tokenizer for completeness if user wants to reload fully
        # But for .pt file request, state_dict is the standard torch way.
        
        print(f"Successfully saved {output_filename}")
        return True
    except Exception as e:
        print(f"Failed to save {model_name}: {e}")
        return False

def main():
    # Define models
    led_model_name = "nsi319/legal-led-base-16384"
    pegasus_model_name = "nsi319/legal-pegasus"
    
    # Define output filenames
    led_output = "legal_led_model.pt"
    pegasus_output = "legal_pegasus_model.pt"
    
    # Save LED
    save_model_as_pt(led_model_name, led_output)
    
    # Save Pegasus
    save_model_as_pt(pegasus_model_name, pegasus_output)
    
    # Verify files exist
    if os.path.exists(led_output):
        print(f"Verified: {led_output} exists ({os.path.getsize(led_output) / (1024*1024):.2f} MB)")
    if os.path.exists(pegasus_output):
        print(f"Verified: {pegasus_output} exists ({os.path.getsize(pegasus_output) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    main()
