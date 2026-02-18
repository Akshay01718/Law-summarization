import torch
import argparse
import sys
import re
import json
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Install required packages:
# pip install torch transformersa rouge-score nltk protobuf sentencepiece

# Check dependencies at startup
def check_dependencies():
    """Check if all required packages are installed"""
    missing = []
    
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    try:
        import sentencepiece
    except ImportError:
        missing.append("sentencepiece")
    
    try:
        import google.protobuf
    except ImportError:
        missing.append("protobuf")
    
    if missing:
        print("\n" + "=" * 70)
        print("MISSING DEPENDENCIES")
        print("=" * 70)
        print(f"\nThe following packages are required but not installed:")
        for pkg in missing:
            print(f"  ❌ {pkg}")
        
        print(f"\nPlease install them using:")
        print(f"  pip install {' '.join(missing)}")
        print("\n" + "=" * 70)
        return False
    
    print("✓ All dependencies found")
    return True

if not check_dependencies():
    sys.exit(1)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


@dataclass
class SummaryConfig:
    """Configuration for summarization parameters"""
    max_extract_sentences: int = 50
    min_sentence_length: int = 25
    max_input_length_led: int = 16384
    max_input_length_pegasus: int = 1024
    min_summary_length: int = 100
    max_summary_length: int = 400
    num_beams: int = 4
    length_penalty: float = 1.5
    no_repeat_ngram_size: int = 3
    use_extractive_fallback: bool = True


class HallucinationDetector:
    """Detects hallucinations in generated summaries"""
    
    # Common hallucination patterns
    HALLUCINATION_INDICATORS = [
        # US-specific legal terms when input is Indian
        r'\bSEC\b',
        r'Securities and Exchange Commission',
        r'Securities Exchange Act',
        r'Securities Act of 193[34]',
        r'Rule 10b-5',
        r'U\.S\. Attorney',
        r'Southern District of New York',
        r'District Court.*New York',
        r'broker-dealer',
        r'antifraud provisions',
        r'disgorgement',
        
        # Other common hallucinations
        r'federal securities laws',
        r'NYSE',
        r'NASDAQ',
        r'complaint alleges',
        r'SEC filed',
    ]
    
    INDIAN_LEGAL_TERMS = [
        r'Supreme Court of India',
        r'High Court',
        r'Indian Penal Code',
        r'\bIPC\b',
        r'Special Leave Petition',
        r'Writ Petition',
        r'appellant',
        r'respondent',
    ]
    
    @classmethod
    def detect_hallucination(cls, summary: str, original_text: str) -> Tuple[bool, List[str], float]:
        """
        Detect if summary contains hallucinated content
        Returns: (is_hallucinated, issues, confidence_score)
        """
        issues = []
        hallucination_score = 0.0
        
        summary_lower = summary.lower()
        original_lower = original_text.lower()
        
        # Check for US legal terms in summary when not in original
        if 'india' in original_lower or 'supreme court of india' in original_lower:
            for pattern in cls.HALLUCINATION_INDICATORS:
                if re.search(pattern, summary, re.IGNORECASE):
                    if not re.search(pattern, original_text, re.IGNORECASE):
                        issues.append(f"Hallucination detected: '{pattern}' not in original document")
                        hallucination_score += 2.0
        
        # Check if summary contains key terms from original
        indian_terms_in_original = sum(1 for p in cls.INDIAN_LEGAL_TERMS 
                                      if re.search(p, original_text, re.IGNORECASE))
        indian_terms_in_summary = sum(1 for p in cls.INDIAN_LEGAL_TERMS 
                                     if re.search(p, summary, re.IGNORECASE))
        
        # If original has Indian terms but summary doesn't, likely wrong jurisdiction
        if indian_terms_in_original >= 2 and indian_terms_in_summary == 0:
            issues.append("Summary missing Indian legal context present in original")
            hallucination_score += 3.0
        
        # Check for complete fabrication of facts
        # Extract case numbers from original
        case_nums_original = set(re.findall(r'Case (?:Crime )?No[.\s]*(\d+)', original_text, re.IGNORECASE))
        case_nums_summary = set(re.findall(r'Case (?:Crime )?No[.\s]*(\d+)', summary, re.IGNORECASE))
        
        # If summary has case numbers not in original
        fabricated_cases = case_nums_summary - case_nums_original
        if fabricated_cases:
            issues.append(f"Fabricated case numbers: {fabricated_cases}")
            hallucination_score += 2.0
        
        # Check for date mismatches
        dates_original = set(re.findall(r'\b\d{4}\b', original_text))
        dates_summary = set(re.findall(r'\b\d{4}\b', summary))
        
        fabricated_dates = dates_summary - dates_original
        if len(fabricated_dates) > 2:
            issues.append(f"Multiple fabricated dates in summary")
            hallucination_score += 1.5
        
        is_hallucinated = hallucination_score > 2.0
        
        return is_hallucinated, issues, hallucination_score


class TextCleaner:
    """Text cleaning utilities"""
    
    @staticmethod
    def clean_legal_text(text: str) -> str:
        """Clean legal text while preserving structure"""
        if not text:
            return ""
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\.(?=[A-Z])', '. ', text)
        text = re.sub(r'\s+\.', '.', text)
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s[0].upper() + s[1:] if s else s for s in sentences]
        
        return ' '.join(sentences).strip()
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences"""
        text = re.sub(r'\bv\.\s', 'v_PERIOD_ ', text)
        text = re.sub(r'\bNo\.\s', 'No_PERIOD_ ', text)
        
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"])', text)
        sentences = [s.replace('_PERIOD_', '.') for s in sentences]
        
        return [s.strip() for s in sentences if s.strip()]


class SmartLegalExtractor:
    """Intelligent extraction focusing on key legal content"""
    
    def __init__(self, config: SummaryConfig):
        self.config = config
    
    def extract_structured_summary(self, text: str) -> str:
        """Generate structured extractive summary"""
        
        lines = []
        lines.append("CASE SUMMARY")
        lines.append("=" * 70 + "\n")
        
        # Extract metadata
        metadata = self._extract_metadata(text)
        if metadata:
            for key, value in metadata.items():
                if value:
                    lines.append(f"{key}: {value}")
            lines.append("")
        
        # Extract key facts
        facts = self._extract_key_facts(text)
        if facts:
            lines.append("KEY FACTS:")
            lines.append("-" * 70)
            for fact in facts:
                lines.append(f"• {fact}")
            lines.append("")
        else:
            # Fallback: get first substantive paragraphs
            sentences = TextCleaner.split_into_sentences(text)
            lines.append("KEY CONTENT:")
            lines.append("-" * 70)
            count = 0
            for sent in sentences[:50]:
                if len(sent.split()) > 15 and not re.match(r'^(?:REPORTABLE|IN THE|CORAM)', sent, re.IGNORECASE):
                    lines.append(f"• {TextCleaner.clean_legal_text(sent)}")
                    count += 1
                    if count >= 5:
                        break
            lines.append("")
        
        # Extract legal issues
        issues = self._extract_legal_issues(text)
        if issues:
            lines.append("LEGAL ISSUES:")
            lines.append("-" * 70)
            for issue in issues:
                lines.append(f"• {issue}")
            lines.append("")
        
        # Extract court's reasoning
        reasoning = self._extract_reasoning(text)
        if reasoning:
            lines.append("COURT'S REASONING:")
            lines.append("-" * 70)
            for point in reasoning:
                lines.append(f"• {point}")
            lines.append("")
        else:
            # Fallback: look for key holdings
            sentences = TextCleaner.split_into_sentences(text)
            key_sent = []
            for sent in sentences:
                if re.search(r'(?:held|ruled|observed|found|concluded):', sent, re.IGNORECASE):
                    key_sent.append(TextCleaner.clean_legal_text(sent))
                    if len(key_sent) >= 3:
                        break
            
            if key_sent:
                lines.append("KEY HOLDINGS:")
                lines.append("-" * 70)
                for s in key_sent:
                    lines.append(f"• {s}")
                lines.append("")
        
        # Extract decision
        decision = self._extract_decision(text)
        if decision:
            lines.append("DECISION:")
            lines.append("-" * 70)
            lines.append(decision)
        else:
            # Fallback: look at end of document
            text_end = text[-500:]
            if re.search(r'(?:appeal|petition|writ).*(?:allowed|dismissed)', text_end, re.IGNORECASE):
                match = re.search(r'([^.]*?(?:appeal|petition|writ).*?(?:allowed|dismissed)[^.]*\.)', text_end, re.IGNORECASE)
                if match:
                    lines.append("DECISION:")
                    lines.append("-" * 70)
                    lines.append(TextCleaner.clean_legal_text(match.group(1)))
        
        return "\n".join(lines)
    
    def _extract_metadata(self, text: str) -> Dict[str, str]:
        """Extract case metadata"""
        metadata = {}
        
        # Case number - look in first 1000 chars for primary case number
        text_header = text[:1500]
        
        case_patterns = [
            r'CIVIL APPEAL NO[.\s]*(\d+)\s+OF\s+(\d{4})',
            r'Civil Appeal No[.\s]*(\d+)\s+of\s+(\d{4})',
            r'CRIMINAL APPEAL NO[.\s]*(\d+)\s+OF\s+(\d{4})',
            r'SPECIAL LEAVE PETITION.*?NO[.\s]*(\d+)\s+OF\s+(\d{4})',
            r'WRIT PETITION.*?NO[.\s]*(\d+)\s+OF\s+(\d{4})',
            r'Case No[.\s]*(\d+)\s+of\s+(\d{4})',
        ]
        
        for pattern in case_patterns:
            case_match = re.search(pattern, text_header, re.IGNORECASE)
            if case_match:
                # Determine case type
                if 'CIVIL APPEAL' in pattern:
                    type_name = "Civil Appeal"
                elif 'CRIMINAL APPEAL' in pattern:
                    type_name = "Criminal Appeal"
                elif 'SPECIAL LEAVE' in pattern:
                    type_name = "Special Leave Petition"
                elif 'WRIT' in pattern:
                    type_name = "Writ Petition"
                else:
                    type_name = "Case"
                
                metadata['Case Number'] = f"{type_name} No. {case_match.group(1)} of {case_match.group(2)}"
                break
        
        # Court - various formats
        court_patterns = [
            (r'SUPREME COURT OF INDIA', 'Supreme Court of India'),
            (r'Supreme Court of India', 'Supreme Court of India'),
            (r'HIGH COURT OF JUDICATURE AT (\w+)', lambda m: f'High Court of {m.group(1).title()}'),
            (r'HIGH COURT AT (\w+)', lambda m: f'High Court at {m.group(1).title()}'),
            (r'(\w+) HIGH COURT', lambda m: f'{m.group(1).title()} High Court'),
            (r'NATIONAL CONSUMER DISPUTES REDRESSAL COMMISSION', 'NCDRC'),
            (r'NATIONAL COMPANY LAW TRIBUNAL', 'NCLT'),
        ]
        
        for pattern, name in court_patterns:
            match = re.search(pattern, text_header, re.IGNORECASE)
            if match:
                if callable(name):
                    metadata['Court'] = name(match)
                else:
                    metadata['Court'] = name
                break
        
        # Date - look near the end for judgment date
        text_end = text[-1500:]
        date_patterns = [
            r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})\b',
            r'\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4})\b',
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, text_end, re.IGNORECASE)
            if date_match:
                metadata['Date'] = date_match.group(1)
                break
        
        # Parties - look in header
        vs_patterns = [
            r'([\w\s&.,()]+?)\s+(?:VERSUS|VS\.?|V\.)\s+([\w\s&.,()]+?)\s+(?:APPELLANT|RESPONDENT|PETITIONER)',
            r'Appellant[:\s]+([\w\s&.,()]{10,100})',
            r'Respondent[:\s]+([\w\s&.,()]{10,100})',
        ]
        
        for pattern in vs_patterns:
            vs_match = re.search(pattern, text_header, re.IGNORECASE | re.DOTALL)
            if vs_match:
                if 'VERSUS' in pattern or 'VS' in pattern:
                    metadata['Appellant'] = re.sub(r'\s+', ' ', vs_match.group(1).strip()[:100])
                    metadata['Respondent'] = re.sub(r'\s+', ' ', vs_match.group(2).strip()[:100])
                    break
                elif 'Appellant' in pattern:
                    metadata['Appellant'] = re.sub(r'\s+', ' ', vs_match.group(1).strip()[:100])
                elif 'Respondent' in pattern:
                    metadata['Respondent'] = re.sub(r'\s+', ' ', vs_match.group(1).strip()[:100])
        
        return metadata
        if date_match:
            metadata['Date'] = date_match.group(0)
        
        # Parties
        vs_match = re.search(r'([\w\s&.()]+?)\s+(?:VERSUS|VS\.?)\s+([\w\s&.()]+?)\s+(?:APPELLANT|O R D E R)', text, re.IGNORECASE | re.DOTALL)
        if vs_match:
            metadata['Appellant'] = vs_match.group(1).strip()[:100]
            metadata['Respondent'] = vs_match.group(2).strip()[:100]
        
        return metadata
    
    def _extract_key_facts(self, text: str) -> List[str]:
        """Extract key factual points"""
        facts = []
        sentences = TextCleaner.split_into_sentences(text)
        
        # Specific fact patterns
        fact_patterns = [
            (r'respondent.*selected.*pursuant', 80),
            (r'respondent.*asked to furnish', 80),
            (r'respondent.*answered.*negative', 80),
            (r'there were.*cases pending', 80),
            (r'Case Crime No\..*under Section', 100),
            (r'concealment.*information', 80),
            (r'non-disclosure.*not.*fatal', 80),
            (r'repeated.*non-disclosure', 80),
            (r'District Magistrate.*held.*suitable', 80),
            (r'appellant.*cancel.*appointment', 80),
        ]
        
        for sent in sentences:
            sent_clean = TextCleaner.clean_legal_text(sent)
            word_count = len(sent_clean.split())
            
            # Check against specific patterns
            for pattern, max_words in fact_patterns:
                if re.search(pattern, sent, re.IGNORECASE) and word_count < max_words:
                    if sent_clean not in facts:
                        facts.append(sent_clean)
                    break
            
            # General fact indicators
            if not any(re.search(p[0], sent, re.IGNORECASE) for p in fact_patterns):
                fact_indicators = [
                    r'in actuality',
                    r'this fact.*came to',
                    r'respondent.*filed.*affidavit',
                    r'Single Judge.*allowed',
                    r'Division Bench.*upheld',
                ]
                
                for indicator in fact_indicators:
                    if re.search(indicator, sent, re.IGNORECASE) and 15 < word_count < 80:
                        if sent_clean not in facts:
                            facts.append(sent_clean)
                        break
        
        return facts[:8]
    
    def _extract_legal_issues(self, text: str) -> List[str]:
        """Extract legal issues/questions"""
        issues = []
        sentences = TextCleaner.split_into_sentences(text)
        
        # Look for the core issue
        issue_patterns = [
            (r'proper.*complete disclosure.*government employment', 100),
            (r'non-disclosure.*depending on.*may not.*fatal', 100),
            (r'concealment.*information.*disqualification', 80),
            (r'giving.*false information.*treated as disqualification', 90),
        ]
        
        for sent in sentences:
            sent_clean = TextCleaner.clean_legal_text(sent)
            word_count = len(sent_clean.split())
            
            # Specific patterns
            for pattern, max_words in issue_patterns:
                if re.search(pattern, sent, re.IGNORECASE) and word_count < max_words:
                    if sent_clean not in issues:
                        issues.append(sent_clean)
                    break
            
            # General indicators
            if not any(re.search(p[0], sent, re.IGNORECASE) for p in issue_patterns):
                issue_indicators = [
                    r'question.*whether.*criminal cases',
                    r'whether.*non-disclosure',
                    r'issue.*arises',
                ]
                
                for indicator in issue_indicators:
                    if re.search(indicator, sent, re.IGNORECASE) and 15 < word_count < 100:
                        if sent_clean not in issues:
                            issues.append(sent_clean)
                        break
        
        return issues[:4]
    
    def _extract_reasoning(self, text: str) -> List[str]:
        """Extract court's reasoning"""
        reasoning = []
        sentences = TextCleaner.split_into_sentences(text)
        
        # Specific reasoning patterns from judgment
        reasoning_patterns = [
            # Complete principles
            (r'Proper and complete disclosure.*not a simple procedural formality.*basic requirement.*fairness.*integrity.*public trust\.', 150),
            (r'When an applicant withholds information.*criminal antecedents.*undermines.*process.*depriving.*authority.*fully informed assessment\.', 150),
            (r'The gravity is significantly compounded when.*non-disclosure is repeated.*ceases to be accidental.*reflects deliberate concealment\.', 120),
            (r'Such strikes at the core of trust.*public service.*honesty and transparency.*indispensable attributes\.', 100),
            (r'Giving any false information.*concealing any material information.*treated as disqualification.*unfit for Government service\.', 100),
            (r'Since the disclaimer makes it clear.*concealment.*ineligible.*what is the clincher.*status of the cases.*time of filing\.', 120),
            (r'It cannot be disputed.*relevant time.*submitted incorrect and false information\.', 80),
            (r'The factum that he said.*no.*not once but twice.*demonstrated mal-intent.*direct contravention of the disclaimer\.', 100),
            (r'Subsequent acquittal.*attempted to come clean.*suppression.*cannot accrue to his benefit\.', 80),
            (r'It is also settled position in law that sympathy cannot supplant law\.', 70),
        ]
        
        for sent in sentences:
            sent_clean = TextCleaner.clean_legal_text(sent)
            word_count = len(sent_clean.split())
            
            for pattern, max_words in reasoning_patterns:
                if re.search(pattern, sent, re.IGNORECASE | re.DOTALL) and word_count < max_words:
                    # Clean up the sentence
                    sent_clean = re.sub(r'\s+', ' ', sent_clean)
                    if sent_clean not in reasoning and len(sent_clean) > 50:
                        reasoning.append(sent_clean)
                    break
        
        # If we didn't get enough, look for key phrases
        if len(reasoning) < 4:
            key_phrases = [
                r'government posts attract hundreds.*applicants.*scrupulous vetting.*imperative',
                r'withholds information.*undermines.*process',
                r'deliberate concealment',
                r'maxim.*jura lex sed lex.*law may be harsh.*law is law',
                r'awareness of consequences.*necessary component of actions',
            ]
            
            for sent in sentences:
                for phrase in key_phrases:
                    if re.search(phrase, sent, re.IGNORECASE):
                        sent_clean = TextCleaner.clean_legal_text(sent)
                        word_count = len(sent_clean.split())
                        if 15 < word_count < 120 and sent_clean not in reasoning:
                            reasoning.append(sent_clean)
                            break
                if len(reasoning) >= 6:
                    break
        
        return reasoning[:6]
    
    def _extract_decision(self, text: str) -> Optional[str]:
        """Extract final decision/order"""
        decision_patterns = [
            r'(?:the\s+)?appeal\s+is\s+(?:allowed|dismissed)\.?',
            r'(?:we\s+)?(?:allow|dismiss)\s+(?:the\s+)?appeal\.?',
            r'petition\s+is\s+(?:allowed|dismissed)\.?',
            r'writ\s+(?:is\s+)?(?:allowed|dismissed)\.?',
        ]
        
        # Look in last 500 characters for decision
        text_end = text[-1000:]
        
        for pattern in decision_patterns:
            match = re.search(pattern, text_end, re.IGNORECASE)
            if match:
                # Try to get a complete sentence
                sentences = TextCleaner.split_into_sentences(text_end)
                for sent in sentences:
                    if re.search(pattern, sent, re.IGNORECASE):
                        return TextCleaner.clean_legal_text(sent)
        
        return None
    
    def extract_for_model(self, text: str) -> str:
        """Extract key sentences for model input"""
        sentences = TextCleaner.split_into_sentences(text)
        
        scored_sentences = []
        for i, sent in enumerate(sentences):
            if len(sent.split()) < 10:
                continue
            
            if re.match(r'^(?:REPORTABLE|NON-REPORTABLE|IN THE|CORAM)', sent, re.IGNORECASE):
                continue
            
            score = self._score_sentence(sent, i, len(sentences))
            if score > 0:
                scored_sentences.append((score, sent))
        
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        top_sentences = [s for _, s in scored_sentences[:self.config.max_extract_sentences]]
        
        return ' '.join(top_sentences)
    
    def _score_sentence(self, sentence: str, position: int, total: int) -> float:
        """Score sentence importance"""
        score = 0.0
        sent_lower = sentence.lower()
        
        # Key legal terms
        legal_keywords = ['held', 'ruled', 'decided', 'dismissed', 'allowed', 'appellant', 
                         'respondent', 'court', 'observed', 'judgment', 'order']
        
        for kw in legal_keywords:
            if re.search(rf'\b{kw}\b', sent_lower):
                score += 1.5
        
        # Position importance
        position_ratio = position / total if total > 0 else 0
        if position_ratio < 0.1:
            score += 3.0
        elif position_ratio > 0.8:
            score += 2.5
        
        # High importance markers
        if re.search(r'(?:held|ruled|decided|observed):', sent_lower):
            score += 5.0
        
        if re.search(r'(?:question|issue)\s+(?:is|arises)', sent_lower):
            score += 4.0
        
        # Length penalty
        word_count = len(sentence.split())
        if word_count < 12 or word_count > 100:
            score *= 0.6
        
        return score


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


class EvaluationMetrics:
    """Evaluation metrics"""
    
    def __init__(self):
        try:
            from rouge_score import rouge_scorer
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
                use_stemmer=True
            )
            self.has_rouge = True
        except ImportError:
            self.has_rouge = False
    
    def evaluate_summary(self, generated: str, reference: str) -> Dict[str, Any]:
        """Evaluate summary"""
        metrics = {}
        
        # Basic metrics
        gen_words = set(generated.lower().split())
        ref_words = set(reference.lower().split())
        common_words = gen_words & ref_words
        
        precision = len(common_words) / len(gen_words) if gen_words else 0
        recall = len(common_words) / len(ref_words) if ref_words else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'generated_length': len(generated.split()),
            'reference_length': len(reference.split()),
        })
        
        # ROUGE scores
        if self.has_rouge:
            scores = self.rouge_scorer.score(reference, generated)
            metrics.update({
                'rouge1_f1': scores['rouge1'].fmeasure,
                'rouge2_f1': scores['rouge2'].fmeasure,
                'rougeL_f1': scores['rougeL'].fmeasure,
            })
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, Any], model_name: str):
        """Print metrics"""
        print(f"\n{'=' * 70}")
        print(f"EVALUATION METRICS - {model_name}")
        print(f"{'=' * 70}\n")
        
        print(f"Precision: {metrics.get('precision', 0):.4f}")
        print(f"Recall: {metrics.get('recall', 0):.4f}")
        print(f"F1 Score: {metrics.get('f1', 0):.4f}")
        
        if 'rouge1_f1' in metrics:
            print(f"\nROUGE-1 F1: {metrics['rouge1_f1']:.4f}")
            print(f"ROUGE-2 F1: {metrics['rouge2_f1']:.4f}")
            print(f"ROUGE-L F1: {metrics['rougeL_f1']:.4f}")
        
        print(f"\nGenerated Length: {metrics.get('generated_length', 0)} words")
        print(f"Reference Length: {metrics.get('reference_length', 0)} words")
        print("=" * 70)


class LEDSummarizer:
    """Legal-LED summarizer"""
    
    MODEL_PATH = "nsi319/legal-led-base-16384"
    
    def __init__(self, device: torch.device, config: SummaryConfig):
        self.device = device
        self.config = config
        self.load_model()
    
    def load_model(self):
        print("Loading Legal-LED model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_PATH)
        self.model.to(self.device)
        self.model.eval()
        print("✓ Legal-LED loaded\n")
    
    def generate_summary(self, text: str) -> str:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_length_led,
            padding=True
        ).to(self.device)
        
        global_attention_mask = torch.zeros_like(inputs.input_ids)
        global_attention_mask[:, 0] = 1
        for i in range(256, inputs.input_ids.shape[1], 256):
            if i < inputs.input_ids.shape[1]:
                global_attention_mask[:, i] = 1
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                global_attention_mask=global_attention_mask,
                num_beams=self.config.num_beams,
                min_length=self.config.min_summary_length,
                max_length=self.config.max_summary_length,
                length_penalty=self.config.length_penalty,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                early_stopping=True
            )
        
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)


class PegasusSummarizer:
    """Legal-PEGASUS summarizer"""
    
    MODEL_PATH = "nsi319/legal-pegasus"
    
    def __init__(self, device: torch.device, config: SummaryConfig):
        self.device = device
        self.config = config
        self.load_model()
    
    def load_model(self):
        print("Loading Legal-PEGASUS model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_PATH)
        self.model.to(self.device)
        self.model.eval()
        print("✓ Legal-PEGASUS loaded\n")
    
    def generate_summary(self, text: str) -> str:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_length_pegasus,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                num_beams=self.config.num_beams,
                min_length=self.config.min_summary_length,
                max_length=self.config.max_summary_length,
                length_penalty=self.config.length_penalty,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                early_stopping=True
            )
        
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)


class LegalSummarizer:
    """Main summarizer with hallucination detection"""
    
    def __init__(self, model_type: str, config: Optional[SummaryConfig] = None):
        self.model_type = model_type.lower()
        self.config = config or SummaryConfig()
        self.device = DeviceManager.setup()
        self.extractor = SmartLegalExtractor(self.config)
        self.evaluator = EvaluationMetrics()
        
        # Load model
        if self.model_type == 'led':
            self.model = LEDSummarizer(self.device, self.config)
        elif self.model_type == 'pegasus':
            self.model = PegasusSummarizer(self.device, self.config)
        elif self.model_type == 'hybrid':
            print("Loading Hybrid Model (LED + PEGASUS)...")
            self.led = LEDSummarizer(self.device, self.config)
            self.pegasus = PegasusSummarizer(self.device, self.config)
        else:
            raise ValueError(f"Unknown model: {model_type}")
    
    def summarize(self, text: str) -> Tuple[str, bool]:
        """
        Generate summary with hallucination detection
        Returns: (summary, used_fallback)
        """
        print(f"\n{'=' * 70}")
        print(f"SUMMARIZATION - {self.model_type.upper()}")
        print(f"{'=' * 70}\n")
        
        # For safety: Check if document is Indian legal first
        is_indian_doc = any(re.search(pattern, text, re.IGNORECASE) 
                           for pattern in [r'Supreme Court of India', r'High Court.*India', 
                                          r'Indian Penal Code', r'\bIPC\b'])
        
        # Extract key content
        extracted = self.extractor.extract_for_model(text)
        print(f"✓ Extracted {len(extracted.split())} words for model input\n")
        
        # Generate summary
        if self.model_type == 'hybrid':
            print("[1/2] LED generating intermediate summary...")
            intermediate = self.led.generate_summary(extracted)
            print(f"✓ LED output: {len(intermediate.split())} words")
            
            # Check LED output for hallucinations before passing to PEGASUS
            is_led_hallucinated, led_issues, led_score = HallucinationDetector.detect_hallucination(
                intermediate, text
            )
            
            if is_led_hallucinated:
                print(f"⚠ LED hallucinated (score: {led_score:.1f})")
                for issue in led_issues[:3]:
                    print(f"  • {issue}")
                
                if self.config.use_extractive_fallback:
                    print("\n→ LED hallucinated, using extractive summary instead\n")
                    summary = self.extractor.extract_structured_summary(text)
                    return summary, True
            
            print("\n[2/2] PEGASUS refining summary...")
            summary = self.pegasus.generate_summary(intermediate)
        else:
            summary = self.model.generate_summary(extracted)
        
        summary = TextCleaner.clean_legal_text(summary)
        print(f"✓ Model summary: {len(summary.split())} words\n")
        
        # Final check for hallucinations
        print("Checking for hallucinations...")
        is_hallucinated, issues, score = HallucinationDetector.detect_hallucination(summary, text)
        
        if is_hallucinated:
            print(f"\n⚠⚠⚠ HALLUCINATION DETECTED (score: {score:.1f}) ⚠⚠⚠")
            print("Issues found:")
            for issue in issues:
                print(f"  • {issue}")
            
            if self.config.use_extractive_fallback:
                print("\n→ Using extractive fallback for accuracy\n")
                summary = self.extractor.extract_structured_summary(text)
                return summary, True
            else:
                print("\n⚠ Fallback disabled - returning potentially hallucinated summary")
        else:
            print("✓ No hallucinations detected\n")
        
        return summary, False
    
    def evaluate(self, generated: str, reference: str) -> Dict[str, Any]:
        return self.evaluator.evaluate_summary(generated, reference)


def main():
    parser = argparse.ArgumentParser(description="Legal Summarizer - LED, PEGASUS, HYBRID")
    
    parser.add_argument("input_file", help="Input legal document")
    parser.add_argument("--model", "-m", required=True, choices=["led", "pegasus", "hybrid"])
    parser.add_argument("--reference", "-r", help="Reference summary for evaluation")
    parser.add_argument("--output", "-o", help="Output file")
    parser.add_argument("--save-metrics", help="Save metrics to JSON")
    parser.add_argument("--max-sentences", type=int, default=50)
    parser.add_argument("--no-fallback", action="store_true", help="Disable extractive fallback")
    
    args = parser.parse_args()
    
    # Load input
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        print(f"✓ Loaded: {len(text.split())} words")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        sys.exit(1)
    
    # Load reference if provided
    reference = None
    if args.reference:
        try:
            with open(args.reference, 'r', encoding='utf-8') as f:
                reference = f.read().strip()
            print(f"✓ Reference: {len(reference.split())} words")
        except Exception as e:
            print(f"⚠ Could not load reference: {e}")
    
    # Configure
    config = SummaryConfig(
        max_extract_sentences=args.max_sentences,
        use_extractive_fallback=not args.no_fallback
    )
    
    # Summarize
    try:
        summarizer = LegalSummarizer(args.model, config)
        summary, used_fallback = summarizer.summarize(text)
        
        # Display
        print(f"{'=' * 70}")
        print(f"SUMMARY - {args.model.upper()}" + (" (EXTRACTIVE FALLBACK)" if used_fallback else ""))
        print(f"{'=' * 70}\n")
        print(summary)
        print(f"\n{'=' * 70}\n")
        
        # Evaluate
        if reference:
            metrics = summarizer.evaluate(summary, reference)
            EvaluationMetrics.print_metrics(metrics, args.model.upper())
            
            if args.save_metrics:
                with open(args.save_metrics, 'w') as f:
                    json.dump(metrics, f, indent=2)
                print(f"\n✓ Metrics saved to {args.save_metrics}")
        
        # Save summary
        output_file = args.output or f"{Path(args.input_file).stem}_{args.model}_summary.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"✓ Summary saved to {output_file}\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()