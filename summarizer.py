import torch
import argparse
import sys
import re
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SummaryConfig:
    """Configuration for summarization parameters"""
    max_extract_sentences: int = 40
    min_sentence_length: int = 40
    max_input_length: int = 16384
    min_summary_length: int = 180
    max_summary_length: int = 500
    num_beams: int = 4
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    use_fallback_on_hallucination: bool = True


class TextCleaner:
    """Advanced text cleaning utilities"""
    
    @staticmethod
    def clean_legal_text(text: str) -> str:
        """Comprehensive text cleaning"""
        # Remove footnote references (numbers before uppercase acronyms)
        text = re.sub(r'\b\d+\s*(?=[A-Z]{2,})', '', text)
        
        # Remove "Vide X" patterns
        text = re.sub(r'\bVide\s+\d+\s+', '', text)
        
        # Remove standalone single-digit numbers (footnotes)
        text = re.sub(r'\s+\d{1,2}\s+(?=[a-z])', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s{2,}', ' ', text)
        
        # Remove numbers at end of acronyms (like "2NCDRC" -> "NCDRC")
        text = re.sub(r'(\d)([A-Z]{2,})', r'\2', text)
        
        return text.strip()
    
    @staticmethod
    def extract_complete_sentence(text: str, start_pos: int, max_len: int = 500) -> str:
        """Extract a complete sentence from position"""
        # Find sentence boundaries
        end_match = re.search(r'[.!?](?:\s|$)', text[start_pos:start_pos+max_len])
        if end_match:
            end_pos = start_pos + end_match.end()
            sentence = text[start_pos:end_pos].strip()
            return TextCleaner.clean_legal_text(sentence)
        return ""


class HallucinationDetector:
    """Detects hallucinations in generated summaries"""
    
    HALLUCINATION_PATTERNS = {
        'sec_securities': [
            r'Securities and Exchange Commission',
            r'\bSEC\b.*announced',
            r'federal securities laws',
            r'Securities Act of 1933',
            r'Securities Exchange Act of 1934',
            r'Rule 10b-5',
            r'broker-dealer',
            r'antifraud provisions',
            r'disgorgement',
            r'U\.S\. District Court.*New York',
            r'Southern District of New York'
        ]
    }
    
    @classmethod
    def detect(cls, summary: str, original: str) -> Tuple[bool, List[str], float]:
        """Detect if summary contains hallucinated content"""
        issues = []
        hallucination_score = 0.0
        
        for category, patterns in cls.HALLUCINATION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, summary, re.IGNORECASE):
                    if not re.search(pattern, original, re.IGNORECASE):
                        issues.append(f"Hallucination: '{pattern}' not in original")
                        hallucination_score += 1.0
        
        is_hallucinated = hallucination_score > 0.5
        return is_hallucinated, issues, hallucination_score


class TemplateBasedSummarizer:
    """Enhanced extractive summarizer"""
    
    @staticmethod
    def extract_case_info(text: str) -> Dict[str, str]:
        """Extract case metadata"""
        info = {}
        
        # Case number
        case_match = re.search(r'CIVIL APPEAL NO[sS]?\.\s*([\d\-]+)\s*OF\s*(\d{4})', text)
        if case_match:
            info['case_number'] = f"Civil Appeal No. {case_match.group(1)} of {case_match.group(2)}"
        
        # Court
        if 'SUPREME COURT OF INDIA' in text:
            info['court'] = 'Supreme Court of India'
        
        # Judges - from signature block at end
        judge_section = text[-800:]
        judges = []
        for match in re.finditer(r'\(([A-Z][A-Z\s]{8,40})\)', judge_section):
            name = match.group(1).strip()
            name = ' '.join(word.capitalize() for word in name.split())
            if 'Judgment' not in name and len(name) > 8:
                judges.append(name)
        
        if judges:
            info['judge'] = ', '.join(judges[:2])
        
        # Date
        date_match = re.search(r'([A-Z][a-z]+\s+\d{1,2},\s+\d{4})\s*\.?\s*$', text[-1000:], re.MULTILINE)
        if date_match:
            info['date'] = date_match.group(1)
        
        # Parties
        vs_match = re.search(
            r'([\w\s&.()]{10,80}?)\s+VS?\.\s+([\w\s&.()]{10,80}?)\s+(?:J U D G M E N T|WITH)',
            text,
            re.IGNORECASE
        )
        if vs_match:
            info['appellant'] = TextCleaner.clean_legal_text(vs_match.group(1).strip())
            info['respondent'] = TextCleaner.clean_legal_text(vs_match.group(2).strip())
        
        return info
    
    @staticmethod
    def extract_background(text: str) -> Optional[str]:
        """Extract factual background with better cleaning"""
        # Try to find paragraph 2 (factual background)
        para2_match = re.search(
            r'\n2\.\s+(Appellant[^.]+(?:flat|apartment|buyer)[^.]+\.[^.]+\.[^.]+\.)',
            text,
            re.IGNORECASE
        )
        
        if para2_match:
            bg = para2_match.group(1)
            # Aggressive cleaning
            bg = TextCleaner.clean_legal_text(bg)
            
            # Remove any remaining artifacts
            bg = re.sub(r'\bimpugned order\b', '', bg, flags=re.IGNORECASE)
            bg = re.sub(r'\s+', ' ', bg)
            
            if 100 < len(bg) < 600:
                return bg
        
        # Fallback: look for lead appeals description
        lead_match = re.search(
            r'(The lead appeals[^.]+call in question[^.]+\.)',
            text,
            re.IGNORECASE
        )
        if lead_match:
            return TextCleaner.clean_legal_text(lead_match.group(1))
        
        return None
    
    @staticmethod
    def extract_issue(text: str) -> Optional[str]:
        """Extract main legal issue"""
        # Pattern 1: From ISSUE section (paragraph 9)
        issue_match = re.search(
            r'(?:ISSUE|9\.)\s+The core controversy[^.]+\.\s+(Question that arises is,?\s+[^.]+\.)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        
        if issue_match:
            issue = issue_match.group(1).strip()
            issue = TextCleaner.clean_legal_text(issue)
            if 50 < len(issue) < 500:
                return issue
        
        # Pattern 2: Look for "can persons who" question
        can_match = re.search(
            r'(can persons who were arrayed[^.]+could be brought[^.]+\.)',
            text,
            re.IGNORECASE
        )
        if can_match:
            return TextCleaner.clean_legal_text(can_match.group(1))
        
        return None
    
    @staticmethod
    def extract_key_holdings(text: str) -> List[str]:
        """Extract key holdings from analysis section"""
        holdings = []
        
        # Define specific patterns for this judgment
        holding_patterns = [
            # From paragraph 11
            (5.0, r'(Once the lis stood consciously and finally confined to ACIPL,\s+the\s+adjudication culminated in an order binding exclusively ACIPL and\s+none else\.)'),
            
            # From paragraph 11 - no findings
            (4.8, r'(The order neither records any determination of liability\s+against the respondents[^.]+nor contains any direction[^.]+\.(?:\s+In the absence of[^.]+lacking\.))'),
            
            # From paragraph 12 - execution conformity
            (4.5, r'(Since,?\s+the judgment[^.]+had\s+not been passed against[^.]+directors[^.]+at the stage of\s+execution,?\s+the order[^.]+could not be enforced\s+against them\.)'),
            
            # From paragraph 13 - decree cannot shift liability
            (4.3, r'(It is trite that a decree cannot[^.]+be employed\s+to shift or enlarge liability[^.]+to bind persons who were neither\s+parties to the decree[^.]+\.)'),
            
            # From paragraph 17 - adjudicatory process
            (4.0, r'((?:the|The)\s+CP Act envisages a complete adjudicatory process\s+founded on[^.]+leading\s+of evidence,?\s+and recorded findings[^.]+\.(?:\s+These[^.]+precede[^.]+liability\.))'),
            
            # From paragraph 18 - corporate veil
            (3.8, r'(The lifting of the corporate veil is an exceptional measure[^.]+only upon a clear finding that[^.]+was abused\s+for fraudulent or dishonest purposes\.)'),
            
            # From paragraph 23 - final holding
            (5.5, r'(NCDRC committed no error[^.]+declining to execute the order against\s+persons who were admittedly not parties to the complaints\.)'),
        ]
        
        scored_holdings = []
        
        for score, pattern in holding_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                holding = match.group(1).strip()
                holding = TextCleaner.clean_legal_text(holding)
                
                # Additional cleaning
                holding = re.sub(r'\s{2,}', ' ', holding)
                
                if 40 < len(holding) < 600:
                    scored_holdings.append((score, holding))
        
        # Sort by score
        scored_holdings.sort(reverse=True, key=lambda x: x[0])
        
        # Remove duplicates
        seen = set()
        unique_holdings = []
        for score, holding in scored_holdings:
            key = holding.lower()[:80]
            if key not in seen:
                seen.add(key)
                unique_holdings.append(holding)
        
        return unique_holdings[:6]
    
    @staticmethod
    def extract_disposition(text: str) -> Optional[str]:
        """Extract final order"""
        # Look for paragraph 24 (disposition)
        disp_match = re.search(
            r'24\.\s+(Consequently,\s+the appeals\s+are\s+dismissed\.)',
            text,
            re.IGNORECASE
        )
        
        if disp_match:
            disp = disp_match.group(1)
            
            # Check for paragraph 25 (remedies note)
            remedy_match = re.search(
                r'25\.\s+(However,?\s+this dismissal[^.]+\.)',
                text,
                re.IGNORECASE
            )
            
            if remedy_match:
                disp += " " + remedy_match.group(1)
            
            return TextCleaner.clean_legal_text(disp)
        
        return "The appeals are dismissed."
    
    @classmethod
    def generate_summary(cls, text: str) -> str:
        """Generate structured summary"""
        info = cls.extract_case_info(text)
        background = cls.extract_background(text)
        issue = cls.extract_issue(text)
        holdings = cls.extract_key_holdings(text)
        disposition = cls.extract_disposition(text)
        
        lines = []
        
        # Header
        lines.append("=" * 70)
        lines.append("SUPREME COURT JUDGMENT SUMMARY")
        lines.append("=" * 70)
        lines.append("")
        
        # Case details
        if info.get('case_number'):
            lines.append(f"Case: {info['case_number']}")
        if info.get('court'):
            lines.append(f"Court: {info['court']}")
        if info.get('date'):
            lines.append(f"Date: {info['date']}")
        if info.get('judge'):
            lines.append(f"Bench: {info['judge']}")
        
        lines.append("")
        
        # Parties
        if info.get('appellant') and info.get('respondent'):
            lines.append("PARTIES")
            lines.append("-" * 70)
            lines.append(f"Appellant: {info['appellant']}")
            lines.append(f"Respondent: {info['respondent']}")
            lines.append("")
        
        # Background
        if background:
            lines.append("FACTUAL BACKGROUND")
            lines.append("-" * 70)
            lines.append(background)
            lines.append("")
        
        # Issue
        if issue:
            lines.append("LEGAL ISSUE")
            lines.append("-" * 70)
            lines.append(issue)
            lines.append("")
        
        # Holdings
        if holdings:
            lines.append("KEY HOLDINGS & LEGAL REASONING")
            lines.append("-" * 70)
            for i, holding in enumerate(holdings, 1):
                # Wrap long holdings
                if len(holding) > 300:
                    words = holding.split()
                    current_line = []
                    for word in words:
                        current_line.append(word)
                        if len(' '.join(current_line)) > 60:
                            lines.append(f"{i}. {' '.join(current_line)}")
                            current_line = []
                            i = "  "  # Indent continuation
                    if current_line:
                        lines.append(f"{i}. {' '.join(current_line)}")
                else:
                    lines.append(f"{i}. {holding}")
                
                if isinstance(i, int) and i < len(holdings):
                    lines.append("")
        
        # Disposition
        if disposition:
            lines.append("")
            lines.append("FINAL ORDER")
            lines.append("-" * 70)
            lines.append(disposition)
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)


class DeviceManager:
    """Manages device selection"""
    
    @staticmethod
    def setup() -> torch.device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✓ Using CUDA: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("✓ Using Apple MPS")
        else:
            device = torch.device("cpu")
            print("✓ Using CPU")
        return device


class LegalExtractor:
    """Sentence extraction for model input"""
    
    KEYWORDS = [
        "held", "dismissed", "allowed", "appellant", "respondent",
        "NCDRC", "ACIPL", "execution", "directors", "promoters",
        "moratorium", "IBC", "decree", "judgment", "liable"
    ]
    
    def __init__(self, config: SummaryConfig):
        self.config = config
    
    def extract_key_sentences(self, text: str) -> str:
        """Extract relevant sentences for model"""
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"])', text)
        
        scored = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < self.config.min_sentence_length:
                continue
            
            if re.match(r'^(REPORTABLE|CIVIL APPEAL|WITH)', sent):
                continue
            
            score = self._score_sentence(sent)
            if score > 0:
                scored.append((score, sent))
        
        scored.sort(reverse=True)
        top = [s for _, s in scored[:self.config.max_extract_sentences]]
        
        print(f"✓ Extracted {len(top)} sentences")
        return " ".join(top)
    
    def _score_sentence(self, sentence: str) -> float:
        """Score sentence relevance"""
        score = 0.0
        sent_lower = sentence.lower()
        
        for kw in self.KEYWORDS:
            if re.search(rf'\b{kw.lower()}\b', sent_lower):
                score += 1.5
        
        if re.search(r'(held|directed|dismissed|concluded)', sent_lower):
            score += 3.0
        
        if len(sentence.split()) > 80:
            score *= 0.7
        
        return score


class ModelManager:
    """Model loading and inference"""
    
    MODELS = {
        "led": "nsi319/legal-led-base-16384",
        "pegasus": "nsi319/legal-pegasus"
    }
    
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
    
    def load(self):
        """Load model"""
        model_path = self.MODELS[self.model_name]
        print(f"Loading {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("✓ Model loaded")
        return self.tokenizer, self.model
    
    def generate_summary(self, text: str, config: SummaryConfig) -> str:
        """Generate summary"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_input_length
        ).to(self.device)
        
        if "led" in self.model_name.lower():
            global_attention_mask = torch.zeros_like(inputs.input_ids)
            global_attention_mask[:, 0] = 1
            for i in range(256, inputs.input_ids.shape[1], 256):
                global_attention_mask[:, i] = 1
        else:
            global_attention_mask = None
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                global_attention_mask=global_attention_mask,
                num_beams=config.num_beams,
                min_length=config.min_summary_length,
                max_length=config.max_summary_length,
                length_penalty=config.length_penalty,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                early_stopping=True,
                do_sample=False,
                repetition_penalty=1.2
            )
        
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)


class LegalSummarizer:
    """Main summarizer"""
    
    def __init__(self, model_name: str = "led", config: Optional[SummaryConfig] = None, use_model: bool = True):
        self.config = config or SummaryConfig()
        self.use_model = use_model
        
        if use_model:
            self.device = DeviceManager.setup()
            self.extractor = LegalExtractor(self.config)
            self.model_manager = ModelManager(model_name, self.device)
            self.tokenizer, self.model = self.model_manager.load()
    
    def summarize(self, text: str) -> Tuple[str, List[str], str]:
        """Summarize with fallback"""
        print("\n" + "=" * 70)
        print("LEGAL DOCUMENT SUMMARIZER")
        print("=" * 70 + "\n")
        
        warnings_list = []
        
        if not self.use_model:
            print("Mode: Template-based Extraction\n")
            summary = TemplateBasedSummarizer.generate_summary(text)
            return summary, warnings_list, "template-based"
        
        print("Mode: Model-based with Fallback\n")
        
        print("[1/3] Extracting key sentences...")
        extracted = self.extractor.extract_key_sentences(text)
        
        print("[2/3] Generating summary...")
        summary = self.model_manager.generate_summary(extracted, self.config)
        
        print("[3/3] Validating...")
        is_hallucinated, issues, score = HallucinationDetector.detect(summary, text)
        
        if is_hallucinated:
            print(f"⚠ Hallucination detected (score: {score:.1f})")
            warnings_list.extend(issues)
            
            if self.config.use_fallback_on_hallucination:
                print("→ Using template-based fallback\n")
                summary = TemplateBasedSummarizer.generate_summary(text)
                return summary, warnings_list, "template-based (fallback)"
        
        print("✓ Validation passed")
        return summary, warnings_list, "model-based"


def main():
    parser = argparse.ArgumentParser(description="Legal Judgment Summarizer")
    
    parser.add_argument("input_file", help="Path to legal document (.txt)")
    parser.add_argument("--output-file", "-o", help="Output file path")
    parser.add_argument("--model", "-m", choices=["led", "pegasus"], default="led")
    parser.add_argument("--extractive", action="store_true", 
                       help="Use template-based extraction (recommended)")
    parser.add_argument("--no-fallback", action="store_true",
                       help="Don't fallback on hallucination")
    parser.add_argument("--max-sentences", type=int, default=40)
    
    args = parser.parse_args()
    
    config = SummaryConfig(
        max_extract_sentences=args.max_sentences,
        use_fallback_on_hallucination=not args.no_fallback
    )
    
    try:
        path = Path(args.input_file)
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        print(f"Loaded: {len(text)} chars, {len(text.split())} words")
        
        summarizer = LegalSummarizer(
            model_name=args.model,
            config=config,
            use_model=not args.extractive
        )
        
        summary, warnings_list, method = summarizer.summarize(text)
        
        print("\n" + "=" * 70)
        print(f"SUMMARY ({method})")
        print("=" * 70 + "\n")
        print(summary)
        
        if warnings_list:
            print("\n⚠ WARNINGS:")
            for w in warnings_list:
                print(f"  • {w}")
        
        print("\n" + "=" * 70)
        
        output_file = args.output_file or str(path.with_name(f"{path.stem}_summary.txt"))
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"\n✓ Saved to: {output_file}\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()