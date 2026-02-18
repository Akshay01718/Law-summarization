import torch
import argparse
import sys
import re
import json
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
import warnings
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Check dependencies
def check_dependencies():
    """Check if all required packages are installed"""
    required = ["torch", "transformers", "sentence_transformers", "sentencepiece", "google.protobuf"]
    missing = []
    
    for pkg in required:
        try:
            if pkg == "google.protobuf":
                import google.protobuf
            elif pkg == "sentence_transformers":
                import sentence_transformers
            else:
                __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        logger.error("=" * 70)
        logger.error("MISSING DEPENDENCIES")
        logger.error("=" * 70)
        logger.error(f"\nThe following packages are required but not installed:")
        for pkg in missing:
            logger.error(f"  ❌ {pkg}")
        logger.error(f"\nPlease install them using:")
        logger.error(f"  pip install {' '.join(missing)}")
        logger.error("=" * 70)
        return False
    
    return True

if not check_dependencies():
    sys.exit(1)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

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
    consistency_threshold: float = 0.5
    max_retries: int = 2


class DeviceManager:
    """Manages device selection and optimization flags"""
    
    @staticmethod
    def get_device_config() -> Dict[str, Any]:
        """Returns device and loading parameters"""
        config = {}
        
        if torch.cuda.is_available():
            config['device'] = torch.device("cuda") 
            config['dtype'] = torch.float32
            config['low_cpu_mem'] = True
            logger.info(f"✓ Using CUDA: {torch.cuda.get_device_name(0)} (FP32)")
        elif torch.backends.mps.is_available():
            config['device'] = torch.device("mps")
            config['dtype'] = torch.float32
            config['low_cpu_mem'] = True
            logger.info("✓ Using Apple MPS")
        else:
            config['device'] = torch.device("cpu")
            config['dtype'] = torch.float32
            config['low_cpu_mem'] = True
            logger.info("✓ Using CPU (Dynamic Quantization Enabled)")
        
        return config


class TextProcessor:
    """Decoupled text processing and cleaning"""
    
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
    def clean_transcript(text: str) -> str:
        """Clean video transcripts (remove timestamps, etc.)"""
        if not text:
            return ""
            
        text = re.sub(r'\[\d{2}:\d{2}(?::\d{2})?\]', '', text) 
        text = re.sub(r'\(\d{2}:\d{2}(?::\d{2})?\)', '', text) 
        text = re.sub(r'Speaker \d+:', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences"""
        if not text or not text.strip():
            return []
            
        # Protect legal abbreviations
        text = re.sub(r'\bv\.\s', 'v_PERIOD_ ', text)
        text = re.sub(r'\bNo\.\s', 'No_PERIOD_ ', text)
        text = re.sub(r'\bvs\.\s', 'vs_PERIOD_ ', text)
        text = re.sub(r'\bDr\.\s', 'Dr_PERIOD_ ', text)
        text = re.sub(r'\bMr\.\s', 'Mr_PERIOD_ ', text)
        text = re.sub(r'\bMrs\.\s', 'Mrs_PERIOD_ ', text)
        text = re.sub(r'\bMs\.\s', 'Ms_PERIOD_ ', text)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', text)
        
        # Restore abbreviations
        sentences = [s.replace('_PERIOD_', '.') for s in sentences]
        
        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Filter out very short sentences (likely fragments)
        sentences = [s for s in sentences if len(s) > 10]
        
        return sentences


class ConsistencyChecker:
    """Checks for semantic contradictions between summary and source"""
    
    MODEL_NAME = 'all-MiniLM-L6-v2'
    
    def __init__(self, device: torch.device):
        logger.info(f"Loading Consistency Checker ({self.MODEL_NAME})...")
        self.model = SentenceTransformer(self.MODEL_NAME, device=str(device))
        logger.info("✓ Consistency Checker loaded")
        
    def check_consistency(self, summary: str, original_text: str, threshold: float = 0.4) -> Tuple[bool, List[str], float]:
        """
        Check if summary is consistent with original text using cosine similarity.
        Returns: (is_consistent, issues, lowest_score)
        """
        # Validate inputs
        if not summary or not summary.strip():
            logger.warning("Empty summary provided to consistency checker")
            return False, ["Empty summary"], 0.0
            
        if not original_text or not original_text.strip():
            logger.warning("Empty original text provided to consistency checker")
            return True, [], 1.0  # Can't check against nothing, assume OK
        
        summary_sentences = TextProcessor.split_into_sentences(summary)
        
        # Handle empty sentence list
        if not summary_sentences:
            logger.warning("No sentences extracted from summary")
            return False, ["No valid sentences in summary"], 0.0
        
        source_sentences = TextProcessor.split_into_sentences(original_text)
        
        if not source_sentences:
            logger.warning("No sentences extracted from source")
            return True, [], 1.0
            
        source_chunks = [' '.join(source_sentences[i:i+3]) for i in range(0, len(source_sentences), 3)]
        
        if not source_chunks:
            return True, [], 1.0
        
        # Encode with error handling
        try:
            summary_embeddings = self.model.encode(summary_sentences, convert_to_tensor=True)
            source_embeddings = self.model.encode(source_chunks, convert_to_tensor=True)
            
            # Validate embeddings shape
            if summary_embeddings.shape[0] == 0 or source_embeddings.shape[0] == 0:
                logger.warning("Empty embeddings generated")
                return True, [], 1.0
                
        except Exception as e:
            logger.warning(f"Error encoding for consistency check: {e}")
            return True, [], 1.0  # Skip check on encoding error
        
        issues = []
        min_score = 1.0
        
        cosine_scores = util.cos_sim(summary_embeddings, source_embeddings)
        
        for i, sent in enumerate(summary_sentences):
            max_sim = torch.max(cosine_scores[i]).item()
            
            if max_sim < min_score:
                min_score = max_sim
            
            if max_sim < threshold:
                issues.append(f"Low consistency ({max_sim:.2f}): '{sent[:50]}...'")
        
        is_consistent = len(issues) == 0
        return is_consistent, issues, min_score


class HallucinationDetector:
    """Detects hallucinations (Rule-based)"""
    
    HALLUCINATION_INDICATORS = [
        r'\bSEC\b', r'Securities and Exchange Commission', r'District Court.*New York',
        r'federal securities laws', r'NYSE', r'NASDAQ'
    ]
    
    INDIAN_LEGAL_TERMS = [
        r'Supreme Court of India', r'High Court', r'Indian Penal Code', r'\bIPC\b'
    ]
    
    @classmethod
    def detect_hallucination(cls, summary: str, original_text: str) -> Tuple[bool, List[str], float]:
        issues = []
        score = 0.0
        
        # Jurisdiction Mismatch
        if 'india' in original_text.lower():
            for pattern in cls.HALLUCINATION_INDICATORS:
                if re.search(pattern, summary, re.IGNORECASE) and not re.search(pattern, original_text, re.IGNORECASE):
                    issues.append(f"Jurisdiction Hallucination: '{pattern}'")
                    score += 2.0
                    
        # Case Number Fabrication
        case_nums_orig = set(re.findall(r'Case (?:Crime )?No[.\s]*(\d+)', original_text, re.IGNORECASE))
        case_nums_summ = set(re.findall(r'Case (?:Crime )?No[.\s]*(\d+)', summary, re.IGNORECASE))
        fabricated = case_nums_summ - case_nums_orig
        if fabricated:
            issues.append(f"Fabricated Case Numbers: {fabricated}")
            score += 2.0

        return score > 0, issues, score


class SmartLegalExtractor:
    """Intelligent extraction focusing on key legal content"""
    
    def __init__(self, config: SummaryConfig, processor: TextProcessor):
        self.config = config
        self.processor = processor
    
    def extract_structured_summary(self, text: str) -> str:
        """Generate structured extractive summary"""
        lines = ["CASE SUMMARY (Extractive Fallback)", "=" * 70 + "\n"]
        
        # Extract metadata
        parties_match = re.search(r'(.+?)\s+(?:vs?\.?|versus)\s+(.+?)(?:\n|AND|$)', text[:2000], re.IGNORECASE)
        if parties_match:
            lines.append(f"PARTIES: {parties_match.group(1).strip()} vs {parties_match.group(2).strip()}")
        
        case_num = re.search(r'(?:Case|Criminal|Civil|Writ|Appeal)\s+No[.\s]*(\d+/\d+|\d+)', text[:2000], re.IGNORECASE)
        if case_num:
            lines.append(f"CASE NUMBER: {case_num.group(0)}")
        
        court = re.search(r'(?:IN THE|BEFORE THE)\s+([A-Z\s]+COURT[A-Z\s]*)', text[:1000], re.IGNORECASE)
        if court:
            lines.append(f"COURT: {court.group(1).strip()}")
        
        lines.append("\n" + "=" * 70)
        
        sentences = self.processor.split_into_sentences(text)
        lines.append("\nKEY CONTENT:")
        lines.append("-" * 70)
        count = 0
        for sent in sentences[:50]:
            if len(sent.split()) > 15 and not re.match(r'^(?:REPORTABLE|IN THE|CORAM)', sent, re.IGNORECASE):
                lines.append(f"• {self.processor.clean_legal_text(sent)}")
                count += 1
                if count >= 8:
                    break
        lines.append("")
        
        # Decision
        text_end = text[-1000:]
        if re.search(r'(?:appeal|petition|writ).*(?:allowed|dismissed)', text_end, re.IGNORECASE):
             match = re.search(r'([^.]*?(?:appeal|petition|writ).*?(?:allowed|dismissed)[^.]*\.)', text_end, re.IGNORECASE)
             if match:
                 lines.append("DECISION:")
                 lines.append("-" * 70)
                 lines.append(self.processor.clean_legal_text(match.group(1)))
                 
        return "\n".join(lines)

    def extract_for_model(self, text: str) -> str:
        """Extract key sentences for model input"""
        sentences = self.processor.split_into_sentences(text)
        
        scored = []
        for i, sent in enumerate(sentences):
            if len(sent.split()) < 8:
                continue
            
            score = 0.0
            lower = sent.lower()
            
            # Keywords
            if any(w in lower for w in ['held', 'ruled', 'concluded', 'observed', 'directed']):
                score += 1.5
            if 'court' in lower:
                score += 0.5
            if any(w in lower for w in ['appeal', 'petition', 'writ']):
                score += 0.8
            if any(w in lower for w in ['allowed', 'dismissed', 'granted', 'rejected']):
                score += 1.0
            
            # Position
            if i < len(sentences) * 0.1:
                score += 1.0
            if i > len(sentences) * 0.9:
                score += 1.0
            
            scored.append((score, sent))
            
        scored.sort(reverse=True, key=lambda x: x[0])
        
        top_sentences = [s for _, s in scored[:self.config.max_extract_sentences]]
        return ' '.join(top_sentences)


class ModelWrapper:
    """Base wrapper for HF models with optimization"""
    
    def __init__(self, model_id: str, device_config: Dict[str, Any], config: SummaryConfig, state_dict_path: Optional[str] = None):
        self.config = config
        self.device = device_config['device']
        self.dtype = device_config['dtype']
        
        logger.info(f"Loading {model_id}...")
        # Use slow tokenizer for better stability with older checkpoints (fixes index out of bounds errors)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        
        # Load model structure (without weights if we have a state dict, but simpler to load with weights then override)
        # Using state_dict is more robust for .pt files
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=device_config.get('low_cpu_mem', False),
            torch_dtype=self.dtype if self.device.type == 'cuda' else torch.float32
        )
        
        if state_dict_path:
            logger.info(f"Loading weights from {state_dict_path}...")
            try:
                state_dict = torch.load(state_dict_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"✓ Weights loaded from {state_dict_path}")
            except Exception as e:
                logger.error(f"Failed to load weights from {state_dict_path}: {e}")
                logger.warning("Using default pretrained weights instead.")

        self.model.to(self.device)
        self.model.eval()
        
        # Dynamic Quantization for CPU
        if self.device.type == 'cpu':
            logger.info("Applying dynamic quantization for CPU...")
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            
        logger.info(f"✓ {model_id} loaded")

    def generate(self, text: str, **kwargs) -> str:
        raise NotImplementedError


class LEDSummarizer(ModelWrapper):
    MODEL_PATH = "nsi319/legal-led-base-16384"
    
    def generate(self, text: str, **kwargs) -> str:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, 
            max_length=self.config.max_input_length_led, padding=True
        ).to(self.device)
        
        global_attention_mask = torch.zeros_like(inputs.input_ids)
        global_attention_mask[:, 0] = 1
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                global_attention_mask=global_attention_mask,
                num_beams=kwargs.get('num_beams', self.config.num_beams),
                min_length=self.config.min_summary_length,
                max_length=self.config.max_summary_length,
                length_penalty=self.config.length_penalty,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                temperature=kwargs.get('temperature', 1.0),
                top_p=kwargs.get('top_p', 1.0),
                do_sample=kwargs.get('do_sample', False)
            )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)


class PegasusSummarizer(ModelWrapper):
    MODEL_PATH = "nsi319/legal-pegasus"
    
    def generate(self, text: str, **kwargs) -> str:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, 
            max_length=self.config.max_input_length_pegasus, padding=True
        ).to(self.device)
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                num_beams=kwargs.get('num_beams', self.config.num_beams),
                min_length=self.config.min_summary_length,
                max_length=self.config.max_summary_length,
                length_penalty=self.config.length_penalty,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                temperature=kwargs.get('temperature', 1.0),
                top_p=kwargs.get('top_p', 1.0),
                do_sample=kwargs.get('do_sample', False)
            )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)


class LegalSummarizer:
    """Main summarizer agent with retry logic"""
    
    def __init__(self, model_type: str, config: Optional[SummaryConfig] = None, led_path: Optional[str] = None, pegasus_path: Optional[str] = None):
        self.model_type = model_type.lower()
        self.config = config or SummaryConfig()
        
        device_config = DeviceManager.get_device_config()
        self.processor = TextProcessor()
        self.extractor = SmartLegalExtractor(self.config, self.processor)
        
        # Load Consistency Checker
        self.checker = ConsistencyChecker(device_config['device'])
        
        # Load Models
        if self.model_type == 'led':
            self.model = LEDSummarizer(LEDSummarizer.MODEL_PATH, device_config, self.config, state_dict_path=led_path)
        elif self.model_type == 'pegasus':
            self.model = PegasusSummarizer(PegasusSummarizer.MODEL_PATH, device_config, self.config, state_dict_path=pegasus_path)
        elif self.model_type == 'hybrid':
            self.led = LEDSummarizer(LEDSummarizer.MODEL_PATH, device_config, self.config, state_dict_path=led_path)
            self.pegasus = PegasusSummarizer(PegasusSummarizer.MODEL_PATH, device_config, self.config, state_dict_path=pegasus_path)
    
    def summarize(self, text: str) -> str:
        logger.info(f"\n{'=' * 50}\nSTARTING SUMMARIZATION ({self.model_type.upper()})\n{'=' * 50}")
        
        # Validate input
        if not text or not text.strip():
            logger.error("Empty or invalid input text")
            return "Error: No input text provided"
        
        logger.info(f"Input document: {len(text.split())} words, {len(text)} characters")
        
        extracted = self.extractor.extract_for_model(text)
        logger.info(f"Extracted for model: {len(extracted.split())} words from {len(text.split())} total words")
        
        if not extracted or len(extracted.split()) < 10:
            logger.warning("Extraction produced insufficient text, using fallback")
            return self.extractor.extract_structured_summary(text)
        
        best_summary = ""
        best_score = 0
        attempt = 0
        
        while attempt <= self.config.max_retries:
            logger.info(f"\n--- Attempt {attempt + 1}/{self.config.max_retries + 1} ---")
            
            params = {}
            if attempt > 0:
                params = {'do_sample': True, 'temperature': max(0.5, 0.7 - (attempt * 0.1)), 'top_p': 0.9}
                logger.info(f"Retrying with parameters: {params}")
            
            # Generate
            try:
                if self.model_type == 'hybrid':
                    logger.info("Generating with LED...")
                    intermediate = self.led.generate(extracted, **params)
                    logger.info(f"LED output: {len(intermediate.split())} words")
                    logger.info("Refining with Pegasus...")
                    summary = self.pegasus.generate(intermediate, **params)
                else:
                    summary = self.model.generate(extracted, **params)
                
                logger.info(f"Generated summary: {len(summary.split())} words")
                
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                if attempt < self.config.max_retries:
                    attempt += 1
                    continue
                else:
                    return self.extractor.extract_structured_summary(text)
            
            summary = self.processor.clean_legal_text(summary)
            
            if not summary or len(summary.split()) < 5:
                logger.warning("Generated summary is too short or empty")
                if attempt < self.config.max_retries:
                    attempt += 1
                    continue
                else:
                    return self.extractor.extract_structured_summary(text)
            
            # Checks
            try:
                hallucinated, h_issues, h_score = HallucinationDetector.detect_hallucination(summary, text)
                consistent, c_issues, c_score = self.checker.check_consistency(summary, extracted, self.config.consistency_threshold)
            except Exception as e:
                logger.warning(f"Quality check failed: {e}")
                # On quality check failure, accept the summary if it looks reasonable
                if len(summary.split()) >= 50:
                    return summary
                consistent = True
                hallucinated = False
                c_score = 0.5
                h_score = 0
            
            # Calculate quality score
            quality_score = c_score - (h_score * 0.1)
            
            if quality_score > best_score:
                best_score = quality_score
                best_summary = summary
            
            if not hallucinated and consistent:
                logger.info(f"✓ Quality Checks Passed (Score: {quality_score:.2f})")
                return summary
            
            logger.warning(f"Quality Check Failed! (Score: {quality_score:.2f})")
            if hallucinated:
                logger.warning(f"Hallucinations detected: {len(h_issues)} issue(s)")
                for issue in h_issues[:3]:
                    logger.warning(f"  - {issue}")
            if not consistent:
                logger.warning(f"Inconsistencies detected: {len(c_issues)} issue(s)")
                for issue in c_issues[:3]:
                    logger.warning(f"  - {issue}")
            
            attempt += 1
        
        if self.config.use_extractive_fallback and best_score < 0.3:
            logger.warning("\nQuality too low. Falling back to extractive summary.")
            return self.extractor.extract_structured_summary(text)
        
        logger.warning(f"\nMaximum retries reached. Returning best effort (Score: {best_score:.2f})")
        return best_summary if best_summary else self.extractor.extract_structured_summary(text)


def main():
    parser = argparse.ArgumentParser(description="Legal Document Summarizer")
    parser.add_argument("input_file", help="Path to input legal document (.txt)")
    parser.add_argument("--model", "-m", required=True, choices=["led", "pegasus", "hybrid"],
                       help="Model to use: led (long docs), pegasus (short), hybrid (both)")
    parser.add_argument("--output", "-o", help="Output file path (default: auto-generated)")
    parser.add_argument("--no-fallback", action="store_true", 
                       help="Disable extractive fallback on failure")
    parser.add_argument("--config", "-c", help="JSON config file for custom parameters")
    parser.add_argument("--led-model-path", help="Path to local .pt file for LED model")
    parser.add_argument("--pegasus-model-path", help="Path to local .pt file for Pegasus model")
    
    args = parser.parse_args()
    
    try:
        # Load input
        input_path = Path(args.input_file)
        if not input_path.exists():
            logger.error(f"Error: File '{args.input_file}' not found")
            sys.exit(1)
            
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        if not text:
            logger.error("Error: Input file is empty")
            sys.exit(1)
        
        logger.info(f"Loaded document: {len(text.split())} words, {len(text)} characters")
        
        # Load config
        config = SummaryConfig(use_extractive_fallback=not args.no_fallback)
        if args.config:
            with open(args.config, 'r') as f:
                custom_config = json.load(f)
                for key, value in custom_config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        
        # Summarize
        summarizer = LegalSummarizer(
            args.model, 
            config, 
            led_path=args.led_model_path, 
            pegasus_path=args.pegasus_model_path
        )
        
        start_time = time.time()
        summary = summarizer.summarize(text)
        end_time = time.time()
        
        # Output
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print(summary)
        print("="*70)
        print(f"\nTime taken: {end_time - start_time:.2f}s")
        print(f"Summary length: {len(summary.split())} words")
        print(f"Compression ratio: {len(text.split()) / max(len(summary.split()), 1):.1f}x")
        
        # Save
        output_file = args.output or f"{input_path.stem}_{args.model}_summary.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        logger.info(f"\n✓ Saved to: {output_file}")
            
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()