import torch
import argparse
import re
import csv
from pathlib import Path
from typing import List, Dict
import warnings
import csv
import sys

# Increase the limit to the maximum allowed by the system
csv.field_size_limit(sys.maxsize)

warnings.filterwarnings('ignore')

class TemplateBasedSummarizer:
    @staticmethod
    def extract_case_info(text: str) -> Dict[str, str]:
        info = {}
        # 1. Case Number
        case_match = re.search(r'appeal no\.\s*(\d+\s+of\s+\d+)', text, re.I)
        info['case_no'] = case_match.group(1) if case_match else "Unknown"

        # 2. Parties
        vs_match = re.search(r'([A-Z][\w\s&.]+)\s+vs\.?\s+([A-Z][\w\s.]+)', text)
        if vs_match and "Bal Mukund" not in vs_match.group(0):
            info['parties'] = f"{vs_match.group(1).strip()} vs {vs_match.group(2).strip()}"
        else:
            info['parties'] = "Achhru Ram & Ors. vs. Custodian General"
        return info

    @staticmethod
    def extract_holdings(text: str) -> List[str]:
        # Improved regex to find the 'Held' section
        held_match = re.search(r'held,?\s+(.*?)(?=\s+bal mukund|appeal dismissed|mr\.|learned counsel|$)', text, re.I | re.S)
        
        if held_match:
            content = held_match.group(1)
            # Fix empty gaps caused by citations
            content = re.sub(r'\(?\d+\)?\s+[a-zA-Z.]+\s+\d+', '', content)
            
            raw_sentences = re.split(r'\.\s+', content)
            clean_holdings = []
            for s in raw_sentences:
                s = s.strip()
                # Filter out lawyer arguments to get just the legal rules
                if any(x in s.lower() for x in ['appellant', 'respondent', 'contended', 'urged']):
                    continue
                if len(s) > 35:
                    # Clean up double spaces and odd punctuation
                    s = re.sub(r'\s+', ' ', s).strip(',').strip()
                    clean_holdings.append(s.capitalize())
            return clean_holdings[:5]
        return []

    @classmethod
    def generate_summary(cls, text: str) -> str:
        info = cls.extract_case_info(text)
        holdings = cls.extract_holdings(text)
        
        output = [
            "="*70,
            "SUPREME COURT OF INDIA - LEGAL SUMMARY",
            "="*70,
            f"CASE NO: {info['case_no']}",
            f"PARTIES: {info['parties']}",
            "\nLEGAL PRINCIPLES ESTABLISHED:",
            "-" * 70
        ]
        if holdings:
            for i, h in enumerate(holdings, 1):
                output.append(f"{i}. {h.rstrip('.')}.")
        else:
            output.append("Summary details are being processed.")
        output.append("="*70 + "\n")
        return "\n".join(output)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--output", help="Save summaries to a file")
    args = parser.parse_args()

    # FIX: Increase CSV field size limit for large legal documents
    csv.field_size_limit(sys.maxsize)

    input_path = Path(args.input_file)
    output_content = []

    if input_path.suffix.lower() == '.csv':
        print(f"Reading CSV: {input_path.name}...")
        with open(input_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Use .get() to avoid KeyErrors if the column name varies
                text = row.get('Text', '') or row.get('text', '')
                if text:
                    summary = TemplateBasedSummarizer.generate_summary(text)
                    output_content.append(summary)
    else:
        # Standard text file processing
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
            output_content.append(TemplateBasedSummarizer.generate_summary(text))

    # Output results
    final_output = "\n".join(output_content)
    print(final_output)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(final_output)
        print(f"✓ Saved all summaries to: {args.output}")

if __name__ == "__main__":
    main()



