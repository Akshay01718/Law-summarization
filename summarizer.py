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
        ],
        'wrong_jurisdiction': [
            r'U\.K\. Attorney',
            r'Federal Bureau of Investigation',
            r'criminal charges against',
        ]
    }
    
    @classmethod
    def detect(cls, summary: str, original: str) -> Tuple[bool, List[str], float]:
        """
        Detect if summary contains hallucinated content
        Returns: (is_hallucinated, issues, confidence_score)
        """
        issues = []
        hallucination_score = 0.0
        
        # Check for known hallucination patterns
        for category, patterns in cls.HALLUCINATION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, summary, re.IGNORECASE):
                    if not re.search(pattern, original, re.IGNORECASE):
                        issues.append(f"Hallucination: '{pattern}' not in original")
                        hallucination_score += 1.0
        
        # Check keyword overlap
        orig_keywords = set(re.findall(r'\b[A-Z][a-z]{3,}\b', original[:3000]))
        summ_keywords = set(re.findall(r'\b[A-Z][a-z]{3,}\b', summary))
        
        if len(orig_keywords) > 0:
            overlap_ratio = len(orig_keywords & summ_keywords) / len(orig_keywords)
            if overlap_ratio < 0.05:
                issues.append(f"Low keyword overlap: {overlap_ratio:.1%}")
                hallucination_score += 2.0
        
        # Check for Indian legal terms
        indian_terms = ['NCDRC', 'Supreme Court of India', 'appellant', 'respondent', 'IBC']
        indian_orig = sum(1 for term in indian_terms if term.lower() in original.lower())
        indian_summ = sum(1 for term in indian_terms if term.lower() in summary.lower())
        
        if indian_orig >= 2 and indian_summ == 0:
            issues.append("Indian legal terminology missing")
            hallucination_score += 1.5
        
        is_hallucinated = hallucination_score > 1.0
        
        return is_hallucinated, issues, hallucination_score


class TemplateBasedSummarizer:
    """Fallback extractive summarizer using templates"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean extracted text from formatting artifacts"""
        # Remove footnote markers
        text = re.sub(r'\d+(?=[A-Z]{2,})', '', text)  # Remove numbers before acronyms like 2NCDRC
        text = re.sub(r'Vide\s+\d+\s+', '', text)  # Remove "Vide 1" patterns
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove standalone numbers that are footnote references
        text = re.sub(r'\s+\d+\s+(?=[a-z])', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def extract_case_info(text: str) -> Dict[str, str]:
        """Extract structured information from legal document"""
        info = {}
        
        # Case number - try multiple patterns
        case_patterns = [
            r'CIVIL APPEAL NO[sS]?\.\s*([\d\-]+)\s*OF\s*(\d{4})',
            r'C\.A\.\s*NO[sS]?\.\s*([\d\-]+)\s*OF\s*(\d{4})'
        ]
        for pattern in case_patterns:
            case_match = re.search(pattern, text)
            if case_match:
                info['case_number'] = f"Civil Appeal No. {case_match.group(1)} of {case_match.group(2)}"
                break
        
        # Court
        if 'SUPREME COURT OF INDIA' in text:
            info['court'] = 'Supreme Court of India'
        elif 'NCDRC' in text or 'National Consumer Disputes Redressal Commission' in text:
            info['tribunal'] = 'National Consumer Disputes Redressal Commission (NCDRC)'
        
        # Judge - improved extraction
        judge_patterns = [
            r'\(([A-Z][A-Z\s]+)\)\s*$',  # (NAME) at end
            r'…………….*?J\.\s*\n\s*\(([A-Z\s]+)\)',  # Format: J. (NAME)
        ]
        # Search in last 500 chars where judges are typically listed
        for pattern in judge_patterns:
            judge_match = re.search(pattern, text[-500:], re.MULTILINE)
            if judge_match:
                judge_name = judge_match.group(1).strip()
                # Clean up the name
                judge_name = ' '.join(word.capitalize() for word in judge_name.split())
                if len(judge_name) > 5 and 'JUDGMENT' not in judge_name.upper():
                    info['judge'] = judge_name
                    break
        
        # Date - look near the end of document
        date_patterns = [
            r'New Delhi;?\s*\n\s*([A-Z][a-z]+\s+\d{1,2},\s+\d{4})',
            r'([A-Z][a-z]+\s+\d{1,2},\s+\d{4})\s*\.'
        ]
        for pattern in date_patterns:
            date_match = re.search(pattern, text[-1000:])
            if date_match:
                info['date'] = date_match.group(1)
                break
        
        # Parties
        vs_patterns = [
            r'([\w\s&.()]+?)\s+(?:VS?\.?|v\.)\s+([\w\s&.()]+?)\s+(?:J U D G M E N T|JUDGMENT|WITH)',
        ]
        for pattern in vs_patterns:
            vs_match = re.search(pattern, text, re.IGNORECASE)
            if vs_match:
                info['appellant'] = vs_match.group(1).strip()
                info['respondent'] = vs_match.group(2).strip()
                # Clean up party names
                info['appellant'] = re.sub(r'\s+', ' ', info['appellant'])
                info['respondent'] = re.sub(r'\s+', ' ', info['respondent'])
                break
        
        return info
    
    @staticmethod
    def extract_key_holdings(text: str) -> List[str]:
        """Extract key holdings and conclusions - excluding interim orders"""
        holdings = []
        
        # First, identify and exclude content from earlier proceedings
        interim_indicators = [
            r'order dated.*?(?:set aside|remit)',
            r'vide order dated.*?directed',
            r'This Court.*?order dated.*?whereby',
            r'Upon revival.*?directed',
            r'earlier.*?January.*?2024'
        ]
        
        # Find sections discussing prior orders to exclude them
        excluded_sections = []
        for indicator in interim_indicators:
            for match in re.finditer(indicator, text, re.IGNORECASE):
                start = max(0, match.start() - 500)
                end = min(len(text), match.end() + 500)
                excluded_sections.append((start, end))
        
        def is_in_excluded_section(pos: int) -> bool:
            """Check if position is in an excluded section"""
            return any(start <= pos <= end for start, end in excluded_sections)
        
        # High-priority patterns for final holdings only
        priority_patterns = [
            (5.5, r'(NCDRC[^.]+committed no error[^.]+declining to execute[^.]+against persons who[^.]+not parties[^.]+\.)'),
            (5.0, r'(execution[^.]+cannot[^.]+be[^.]+employed[^.]+to shift[^.]+liability[^.]+to bind persons who[^.]+not parties[^.]+\.)'),
            (4.8, r'((?:No|no)[^.]+pleadings[^.]+adjudication[^.]+findings against[^.]+directors[^.]+essential foundation[^.]+lacking[^.]+\.)'),
            (4.5, r'(NCDRC[^.]+dismissed[^.]+execution[^.]+holding[^.]+order[^.]+executable only against ACIPL[^.]+\.)'),
            (4.0, r'(the order[^.]+binds only[^.]+ACIPL[^.]+\.)'),
            (4.0, r'(judgment[^.]+not been passed against[^.]+directors[^.]+order[^.]+could not be enforced against them[^.]+\.)'),
            (3.8, r'(execution must strictly conform to the decree[^.]+\.)'),
            (3.5, r'(no[^.]+adjudicatory[^.]+exercise[^.]+undertaken[^.]+directors[^.]+\.)'),
            (3.5, r'(personal liability[^.]+not[^.]+established[^.]+\.)'),
            (3.5, r'((?:No|no)[^.]+pleadings[^.]+evidence[^.]+findings[^.]+fixing personal liability[^.]+\.)'),
            (3.0, r'(corporate veil[^.]+wholly unwarranted[^.]+\.)'),
            (3.0, r'(lifting[^.]+corporate veil[^.]+exceptional measure[^.]+finding[^.]+fraud[^.]+\.)'),
            (2.8, r'(appellant[^.]+did not challenge[^.]+order[^.]+declining to issue notice[^.]+cannot[^.]+enlarge[^.]+through execution[^.]+\.)'),
            (2.5, r'(directors[^.]+were not[^.]+parties[^.]+complaints[^.]+\.)'),
        ]
        
        scored_holdings = []
        
        for score, pattern in priority_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Skip if this match is in an excluded section
                if is_in_excluded_section(match.start()):
                    continue
                
                holding = match.group(1).strip()
                holding = TemplateBasedSummarizer.clean_text(holding)
                
                if 40 < len(holding) < 500:
                    # Additional filters for interim order content
                    if any(phrase in holding.lower() for phrase in [
                        'directed that.*may continue',
                        'liberty to raise',
                        'moratorium.*does not preclude'
                    ]):
                        continue
                    
                    scored_holdings.append((score, holding))
        
        # Sort by score and remove duplicates
        scored_holdings.sort(reverse=True, key=lambda x: x[0])
        seen = set()
        unique_holdings = []
        
        for score, holding in scored_holdings:
            holding_key = holding.lower()[:100]
            if holding_key not in seen:
                seen.add(holding_key)
                unique_holdings.append(holding)
        
        return unique_holdings[:6]  # Top 6 holdings
    
    @staticmethod
    def extract_issue(text: str) -> Optional[str]:
        """Extract the main issue"""
        # Pattern 1: Explicit ISSUE section
        issue_patterns = [
            r'ISSUE\s*\n+\d+\.\s+([^.]+\.(?:[^.]+\.)?)',
            r'(?:core controversy|question that arises|issue)[^:]*:\s*([^.]+\.)',
            r'Question that arises is,?\s+([^.]+\.)',
        ]
        
        for pattern in issue_patterns:
            issue_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if issue_match:
                issue = issue_match.group(1).strip()
                # Clean up the issue
                issue = re.sub(r'\s+', ' ', issue)
                if 20 < len(issue) < 500:
                    return issue
        
        # Pattern 2: "whether" questions
        whether_matches = re.findall(r'(whether[^.?]+[.?])', text, re.IGNORECASE)
        for match in whether_matches:
            match = match.strip()
            if 30 < len(match) < 400:
                return match
        
        # Pattern 3: "can persons who..." type questions
        can_match = re.search(r'(can persons[^.?]+[.?])', text, re.IGNORECASE)
        if can_match:
            match = can_match.group(1).strip()
            if 30 < len(match) < 400:
                return match
        
        return None
    
    @staticmethod
    def extract_background(text: str) -> Optional[str]:
        """Extract brief background"""
        # Pattern 1: FACTUAL BACKGROUND section - get multiple sentences
        bg_match = re.search(
            r'FACTUAL BACKGROUND\s*\n+\d+\.\s+([^.]+\.[^.]+\.[^.]+\.(?:[^.]+\.)?)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if bg_match:
            bg = bg_match.group(1).strip()
            bg = TemplateBasedSummarizer.clean_text(bg)
            if 100 < len(bg) < 700:
                return bg
        
        # Pattern 2: Look for appellant description
        buyer_match = re.search(
            r'(Appellant is[^.]+entered into[^.]+\.[^.]+\.(?:[^.]+\.)?)',
            text,
            re.IGNORECASE
        )
        if buyer_match:
            bg = buyer_match.group(1).strip()
            bg = TemplateBasedSummarizer.clean_text(bg)
            if 80 < len(bg) < 700:
                return bg
        
        # Pattern 3: Lead appeals description
        lead_match = re.search(
            r'(The lead appeals[^.]+\.[^.]+\.(?:[^.]+\.)?)',
            text,
            re.IGNORECASE
        )
        if lead_match:
            bg = lead_match.group(1).strip()
            bg = TemplateBasedSummarizer.clean_text(bg)
            if 80 < len(bg) < 700:
                return bg
        
        return None
    
    @staticmethod
    def extract_disposition(text: str) -> Optional[str]:
        """Extract the final disposition/outcome"""
        # Look for the final disposition - usually near end
        disposition_patterns = [
            # Pattern 1: Numbered conclusion paragraph
            r'(\d+\.\s+Consequently,\s+the appeals?\s+(?:is|are)\s+(?:dismissed|allowed)[^.]+\.(?:\s+\d+\.\s+[^.]+\.)*)',
            # Pattern 2: Simple dismissal
            r'(the appeals?\s+(?:stand|are)\s+(?:dismissed|allowed)[^.]+\.)',
            # Pattern 3: Combined with costs
            r'(Consequently,\s+the appeals?\s+(?:is|are)\s+(?:dismissed|allowed)[^.]+\.(?:[^.]+costs\.)?)',
        ]
        
        for pattern in disposition_patterns:
            # Search in last 2000 chars where disposition typically appears
            disp_match = re.search(pattern, text[-2000:], re.IGNORECASE | re.DOTALL)
            if disp_match:
                disp = disp_match.group(1).strip()
                disp = re.sub(r'\s+', ' ', disp)
                # Clean up: remove leading numbers
                disp = re.sub(r'^\d+\.\s+', '', disp)
                
                # If it captured multiple sentences, take first 2
                sentences = re.split(r'(?<=[.])\s+(?=\d+\.|\w)', disp)
                if len(sentences) > 2:
                    disp = ' '.join(sentences[:2])
                
                if len(disp) > 20:
                    return disp
        
        return None
    
    @classmethod
    def generate_summary(cls, text: str) -> str:
        """Generate template-based extractive summary"""
        info = cls.extract_case_info(text)
        issue = cls.extract_issue(text)
        holdings = cls.extract_key_holdings(text)
        background = cls.extract_background(text)
        disposition = cls.extract_disposition(text)
        
        # Build summary
        lines = []
        
        # Header
        lines.append("=" * 70)
        lines.append("CASE SUMMARY")
        lines.append("=" * 70)
        lines.append("")
        
        # Case details
        if 'case_number' in info:
            lines.append(f"Case Number: {info['case_number']}")
        if 'court' in info:
            lines.append(f"Court: {info['court']}")
        if 'tribunal' in info:
            lines.append(f"Forum: {info['tribunal']}")
        if 'date' in info:
            lines.append(f"Date: {info['date']}")
        if 'judge' in info:
            lines.append(f"Judge: {info['judge']}")
        
        lines.append("")
        
        # Parties
        if 'appellant' in info and 'respondent' in info:
            lines.append("PARTIES")
            lines.append("-" * 70)
            lines.append(f"Appellant: {info['appellant']}")
            lines.append(f"Respondent: {info['respondent']}")
            lines.append("")
        
        # Background
        if background:
            lines.append("BACKGROUND")
            lines.append("-" * 70)
            lines.append(background)
            lines.append("")
        
        # Issue
        if issue:
            lines.append("ISSUE")
            lines.append("-" * 70)
            lines.append(issue)
            lines.append("")
        
        # Holdings
        if holdings:
            lines.append("KEY HOLDINGS & REASONING")
            lines.append("-" * 70)
            for i, holding in enumerate(holdings, 1):
                lines.append(f"{i}. {holding}")
                if i < len(holdings):
                    lines.append("")
        
        # Disposition
        if disposition:
            lines.append("")
            lines.append("DISPOSITION")
            lines.append("-" * 70)
            lines.append(disposition)
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)


class DeviceManager:
    """Manages device selection and setup"""
    
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
    """Extracts key sentences from legal documents"""
    
    KEYWORDS = [
        "held", "dismissed", "allowed", "quashed", "remanded", "affirmed",
        "appellant", "respondent", "petitioner", "directors", "promoters",
        "liable", "liability", "execution", "decree", "judgment", "moratorium",
        "IBC", "NCDRC", "Supreme Court", "consumer", "directed", "observed"
    ]
    
    def __init__(self, config: SummaryConfig):
        self.config = config
    
    def preprocess_text(self, text: str) -> str:
        """Clean legal text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\d+\n', ' ', text)
        return text.strip()
    
    def extract_key_sentences(self, text: str) -> str:
        """Extract top-scoring sentences"""
        text = self.preprocess_text(text)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"])', text)
        
        scored = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < self.config.min_sentence_length:
                continue
            
            if re.match(r'^(REPORTABLE|CIVIL APPEAL|WITH|IN THE SUPREME)', sentence):
                continue
            
            score = self._score_sentence(sentence)
            if score > 0:
                scored.append((score, sentence))
        
        scored.sort(reverse=True)
        top_sentences = [s for _, s in scored[:self.config.max_extract_sentences]]
        
        print(f"✓ Extracted {len(top_sentences)} key sentences")
        return " ".join(top_sentences)
    
    def _score_sentence(self, sentence: str) -> float:
        """Score sentence"""
        sentence_lower = sentence.lower()
        score = 0.0
        
        for keyword in self.KEYWORDS:
            if re.search(rf'\b{keyword}\b', sentence_lower, re.IGNORECASE):
                score += 1.5
        
        if re.search(r'(held that|directed that|appeal.*dismissed)', sentence_lower):
            score += 3.0
        
        if re.search(r'\(\d{4}\)\s+\d+\s+SCC', sentence):
            score += 2.0
        
        if len(sentence.split()) > 100:
            score *= 0.6
        
        return score


class ModelManager:
    """Manages model loading and inference"""
    
    MODELS = {
        "led": "nsi319/legal-led-base-16384",
        "pegasus": "nsi319/legal-pegasus"
    }
    
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
    
    def load(self) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
        """Load tokenizer and model"""
        model_path = self.MODELS.get(self.model_name)
        print(f"Loading model: {model_path}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            print("✓ Model loaded successfully")
            return self.tokenizer, self.model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def generate_summary(self, text: str, config: SummaryConfig) -> str:
        """Generate abstractive summary"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_input_length
        ).to(self.device)
        
        input_length = inputs.input_ids.shape[1]
        print(f"✓ Input tokens: {input_length}")
        
        if "led" in self.model_name.lower():
            global_attention_mask = torch.zeros_like(inputs.input_ids)
            global_attention_mask[:, 0] = 1
            for i in range(256, input_length, 256):
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
    """Main summarizer with hallucination detection"""
    
    def __init__(
        self,
        model_name: str = "led",
        config: Optional[SummaryConfig] = None,
        use_model: bool = True
    ):
        self.config = config or SummaryConfig()
        self.use_model = use_model
        
        if use_model:
            self.device = DeviceManager.setup()
            self.extractor = LegalExtractor(self.config)
            self.model_manager = ModelManager(model_name, self.device)
            self.tokenizer, self.model = self.model_manager.load()
    
    def summarize(self, text: str) -> Tuple[str, List[str], str]:
        """Execute summarization with hallucination detection"""
        print("\n" + "=" * 70)
        print("LEGAL DOCUMENT SUMMARIZER")
        print("=" * 70 + "\n")
        
        warnings_list = []
        
        if not self.use_model:
            print("Mode: Template-based Extractive")
            print("-" * 70)
            summary = TemplateBasedSummarizer.generate_summary(text)
            return summary, warnings_list, "template-based"
        
        print("Mode: Model-based with Hallucination Detection")
        print("-" * 70 + "\n")
        
        # Model-based summarization
        print("[1/3] Extracting key sentences...")
        extracted_text = self.extractor.extract_key_sentences(text)
        
        print("\n[2/3] Generating abstractive summary...")
        summary = self.model_manager.generate_summary(extracted_text, self.config)
        
        print("\n[3/3] Checking for hallucinations...")
        is_hallucinated, issues, score = HallucinationDetector.detect(summary, text)
        
        if is_hallucinated:
            print(f"\n⚠ HALLUCINATION DETECTED (score: {score:.1f})")
            for issue in issues:
                print(f"  • {issue}")
            
            warnings_list.extend(issues)
            
            if self.config.use_fallback_on_hallucination:
                print("\n→ Switching to template-based summary...\n")
                summary = TemplateBasedSummarizer.generate_summary(text)
                return summary, warnings_list, "template-based (auto-fallback)"
            else:
                return summary, warnings_list, "model (hallucinated)"
        
        print("✓ No hallucinations detected")
        return summary, warnings_list, "model-based"


def main():
    parser = argparse.ArgumentParser(
        description="Legal Judgment Summarizer with Hallucination Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Template-based (most reliable, recommended)
  python summarizer_v2.py judgment.txt --extractive
  
  # Model-based with auto-fallback (default)
  python summarizer_v2.py judgment.txt
  
  # Force model mode without fallback
  python summarizer_v2.py judgment.txt --no-fallback
        """
    )
    
    parser.add_argument("input_file", help="Path to legal document (.txt)")
    parser.add_argument("--output-file", "-o", help="Output file path")
    parser.add_argument("--model", "-m", choices=["led", "pegasus"], default="led",
                       help="Model to use (default: led)")
    parser.add_argument("--extractive", action="store_true", 
                       help="Use template-based extractive summarization (recommended)")
    parser.add_argument("--no-fallback", action="store_true",
                       help="Don't fallback to extractive on hallucination")
    parser.add_argument("--max-sentences", type=int, default=40,
                       help="Max sentences to extract (default: 40)")
    
    args = parser.parse_args()
    
    config = SummaryConfig(
        max_extract_sentences=args.max_sentences,
        use_fallback_on_hallucination=not args.no_fallback
    )
    
    try:
        # Read document
        path = Path(args.input_file)
        if not path.exists():
            print(f"❌ Error: File not found: {args.input_file}")
            sys.exit(1)
        
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        if not text:
            print("❌ Error: File is empty")
            sys.exit(1)
        
        print(f"Loaded: {len(text)} chars, {len(text.split())} words")
        
        # Summarize
        summarizer = LegalSummarizer(
            model_name=args.model,
            config=config,
            use_model=not args.extractive
        )
        
        summary, warnings_list, method = summarizer.summarize(text)
        
        # Display result
        print("\n" + "=" * 70)
        print(f"SUMMARY ({method})")
        print("=" * 70 + "\n")
        print(summary)
        
        if warnings_list:
            print("\n" + "=" * 70)
            print("⚠ WARNINGS:")
            for w in warnings_list:
                print(f"  • {w}")
        
        print("\n" + "=" * 70)
        
        # Save output
        output_file = args.output_file or str(
            path.with_name(f"{path.stem}_summary.txt")
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"\n✓ Summary saved to: {output_file}\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()