"""
Diagnostic script - paste the output so the regex patterns can be fixed.
Usage: py -3.11 inspect_pdf.py "D:\A1-mini\evaluation(180-words)\bert.pdf"
"""
import sys
import pdfplumber

pdf_path = sys.argv[1] if len(sys.argv) > 1 else input("Enter path to one PDF: ").strip('"')

print(f"\n{'='*60}")
print(f"FILE: {pdf_path}")
print(f"{'='*60}\n")

with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        print(f"--- PAGE {i+1} ---")
        print(text)
        print()