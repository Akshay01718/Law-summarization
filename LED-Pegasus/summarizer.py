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

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


def check_dependencies():
    required = ["torch", "transformers", "sentence_transformers", "sentencepiece", "google.protobuf"]
    missing = []
    for pkg in required:
        try:
            if pkg == "google.protobuf": import google.protobuf
            elif pkg == "sentence_transformers": import sentence_transformers
            else: __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error(f"Missing packages: {missing}\nInstall: pip install {' '.join(missing)}")
        return False
    return True


if not check_dependencies():
    sys.exit(1)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util


@dataclass
class SummaryConfig:
    max_extract_sentences: int = 18
    min_sentence_length: int = 25
    max_input_length_led: int = 4096
    max_input_length_pegasus: int = 512
    min_summary_length: int = 60
    max_summary_length: int = 400
    num_beams: int = 3
    length_penalty: float = 2.5
    no_repeat_ngram_size: int = 4
    use_extractive_fallback: bool = True
    consistency_threshold: float = 0.35
    max_retries: int = 2


class DeviceManager:
    @staticmethod
    def get_device_config() -> Dict[str, Any]:
        config = {}
        if torch.cuda.is_available():
            config.update(device=torch.device("cuda"), dtype=torch.float32, low_cpu_mem=True)
            logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            config.update(device=torch.device("mps"), dtype=torch.float32, low_cpu_mem=True)
            logger.info("Using Apple MPS")
        else:
            config.update(device=torch.device("cpu"), dtype=torch.float32, low_cpu_mem=True)
            logger.info("Using CPU")
        return config


class TextProcessor:
    @staticmethod
    def clean_legal_text(text: str) -> str:
        if not text: return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\.(?=[A-Z])', '. ', text)
        text = re.sub(r'\s+\.', '.', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s[0].upper() + s[1:] if s else s for s in sentences]
        return ' '.join(sentences).strip()

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        if not text or not text.strip(): return []
        for abbr, repl in [
            (r'\bv\.\s', 'v_P_ '), (r'\bNo\.\s', 'No_P_ '),
            (r'\bvs\.\s', 'vs_P_ '), (r'\bDr\.\s', 'Dr_P_ '),
            (r'\bMr\.\s', 'Mr_P_ '), (r'\bMrs\.\s', 'Mrs_P_ '),
        ]:
            text = re.sub(abbr, repl, text)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', text)
        sentences = [s.replace('_P_', '.').strip() for s in sentences]
        return [s for s in sentences if len(s) > 10]


class JurisdictionCorrector:
    """Fix US legal hallucinations in generated output before quality checks."""

    SUBSTITUTIONS = [
        (r'United States District Court(?:\s+for\s+the\s+[\w\s]+District\s+of\s+[\w\s]+)?',
         'Supreme Court of India'),
        (r'U\.S\.\s+District Court(?:\s+for\s+the\s+[\w\s]+)?', 'Supreme Court of India'),
        (r'Federal\s+(?:District\s+)?Court(?:\s+of\s+[\w\s]+)?', 'High Court'),
        (r'Circuit Court of Appeals', 'Division Bench'),
        (r'Court of Appeals(?:\s+for\s+the\s+[\w\s]+Circuit)?', 'High Court'),
        (r'U\.S\.\s+Supreme Court', 'Supreme Court of India'),
        (r'(?:the\s+)?Honorable\s+(?:Judge\s+)?([A-Z][a-z]+(?:\s+[A-Z]\.?\s+[A-Z][a-z]+)?)',
         r"Hon'ble Justice \1"),
        (r'\bHonorable\b', "Hon'ble"),
        (r'\bentered\s+(?:a\s+)?(?:final\s+)?judgment\b', 'passed the final order'),
        (r'\bentered\s+(?:an?\s+)?order\b', 'passed an order'),
        (r'\bplaintiff\b', 'appellant'), (r'\bPlaintiff\b', 'Appellant'),
        (r'\bdefendant\b', 'respondent'), (r'\bDefendant\b', 'Respondent'),
        (r'\bclass action\b', 'writ petition'),
        (r'\bindictment\b', 'charge sheet'),
        (r'\bdistrict attorney\b', 'public prosecutor'),
        (r'\bfelony\b', 'cognisable offence'),
        (r'Central District of New Delhi', 'New Delhi'),
        (r'Southern District of New Delhi', 'New Delhi'),
        (r'District of Columbia', 'Delhi'),
        (r'\bSEC\b(?!\s+(?:act|section))', 'SEBI'),
        (r'Securities and Exchange Commission(?!\s+of India)',
         'Securities and Exchange Board of India'),
        (r'\bNASDAQ\b', 'NSE'), (r'\bNYSE\b', 'BSE'),
        (r'\bFBI\b', 'CBI'),
    ]

    UNSALVAGEABLE = [
        r'permanently\s+enjoin',
        r'enjoins?\s+.{0,30}(?:officer|director)',
        r'disgorgement',
        r'civil\s+penalty',
        r'for\s+the\s+District\s+of',
        r'United States(?!\s+District\s+Court)',
        r'New\s+York\s+(?:court|district)',
        r'Washington\s*,?\s*D\.?C\.?',
    ]

    def correct(self, text: str) -> Tuple[str, int]:
        if not text: return text, 0
        n = 0
        for pattern, replacement in self.SUBSTITUTIONS:
            new = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            if new != text:
                n += 1
                text = new
        sentences = re.split(r'(?<=[.!?])\s+', text)
        kept = [s for s in sentences
                if not any(re.search(p, s, re.IGNORECASE) for p in self.UNSALVAGEABLE)]
        text = ' '.join(kept).strip()
        parts = re.split(r'(?<=[.!?])\s+', text)
        text = ' '.join(p[0].upper() + p[1:] if p else p for p in parts)
        return text, n


class ConsistencyChecker:
    MODEL_NAME = 'all-MiniLM-L6-v2'

    def __init__(self, device: torch.device):
        logger.info(f"Loading consistency checker ({self.MODEL_NAME})...")
        self.model = SentenceTransformer(self.MODEL_NAME, device=str(device))
        logger.info("Consistency checker ready")

    def check_consistency(self, summary: str, original_text: str,
                          threshold: float = 0.35) -> Tuple[bool, List[str], float]:
        if not summary or not summary.strip(): return False, ["Empty summary"], 0.0
        if not original_text or not original_text.strip(): return True, [], 1.0
        sum_sents = TextProcessor.split_into_sentences(summary)
        if not sum_sents: return False, ["No valid sentences"], 0.0
        src_sents = TextProcessor.split_into_sentences(original_text)
        if not src_sents: return True, [], 1.0
        src_chunks = [' '.join(src_sents[i:i+3]) for i in range(0, len(src_sents), 3)]
        try:
            s_emb = self.model.encode(sum_sents, convert_to_tensor=True)
            r_emb = self.model.encode(src_chunks, convert_to_tensor=True)
            if s_emb.shape[0] == 0 or r_emb.shape[0] == 0: return True, [], 1.0
        except Exception as e:
            logger.warning(f"Consistency encoding error: {e}")
            return True, [], 1.0
        issues, min_score = [], 1.0
        scores = util.cos_sim(s_emb, r_emb)
        for i, sent in enumerate(sum_sents):
            mx = torch.max(scores[i]).item()
            if mx < min_score: min_score = mx
            if mx < threshold: issues.append(f"Low sim ({mx:.2f}): '{sent[:50]}...'")
        return len(issues) == 0, issues, min_score


class HallucinationDetector:
    HALLUCINATION_INDICATORS = [
        r'\bSEC\b', r'Securities and Exchange Commission',
        r'District Court.*New York', r'federal securities laws',
        r'NYSE', r'NASDAQ', r'United States District Court',
        r'permanently enjoin', r'\bplaintiff\b', r'\bdefendant\b',
    ]

    @classmethod
    def detect_hallucination(cls, summary: str, original_text: str) -> Tuple[bool, List[str], float]:
        issues, score = [], 0.0
        if 'india' in original_text.lower():
            for pattern in cls.HALLUCINATION_INDICATORS:
                if (re.search(pattern, summary, re.IGNORECASE)
                        and not re.search(pattern, original_text, re.IGNORECASE)):
                    issues.append(f"Hallucination: '{pattern}'")
                    score += 2.0
        orig_nums = set(re.findall(r'Case (?:Crime )?No[.\s]*(\d+)', original_text, re.IGNORECASE))
        summ_nums = set(re.findall(r'Case (?:Crime )?No[.\s]*(\d+)', summary, re.IGNORECASE))
        fab = summ_nums - orig_nums
        if fab:
            issues.append(f"Fabricated case numbers: {fab}")
            score += 2.0
        return score > 0, issues, score


class SmartLegalExtractor:
    def __init__(self, config: SummaryConfig, processor: TextProcessor):
        self.config = config
        self.processor = processor

    def extract_structured_summary(self, text: str) -> str:
        lines = []
        parties = re.search(r'(.+?)\s+(?:vs?\.?|versus)\s+(.+?)(?:\n|AND|$)', text[:2000], re.IGNORECASE)
        if parties:
            lines.append(f"Parties: {parties.group(1).strip()} vs {parties.group(2).strip()}.")
        case_num = re.search(r'(?:Case|Criminal|Civil|Writ|Appeal)\s+No[.\s]*(\d+/\d+|\d+)', text[:2000], re.IGNORECASE)
        if case_num:
            lines.append(f"Case number: {case_num.group(0)}.")
        court = re.search(r'(?:IN THE|BEFORE THE)\s+([A-Z\s]+COURT[A-Z\s]*)', text[:1000], re.IGNORECASE)
        if court:
            lines.append(f"Court: {court.group(1).strip()}.")
        sentences = self.processor.split_into_sentences(text)
        count = 0
        for sent in sentences[:60]:
            if (len(sent.split()) > 15
                    and not re.match(r'^(?:REPORTABLE|NON.REPORTABLE|IN THE|CORAM|BEFORE)', sent, re.IGNORECASE)
                    and not re.search(r'\.{2,}|…|APPELLANT\(S\)|RESPONDENT\(S\)', sent, re.IGNORECASE)):
                lines.append(self.processor.clean_legal_text(sent))
                count += 1
                if count >= 6:
                    break
        m = re.search(
            r'([^.]*?(?:appeal|petition|writ).*?(?:allowed|dismissed|granted|rejected)[^.]*\.)',
            text[-1500:], re.IGNORECASE
        )
        if m:
            lines.append(self.processor.clean_legal_text(m.group(1)))
        return ' '.join(lines)

    def extract_for_model(self, text: str) -> str:
        sentences = self.processor.split_into_sentences(text)
        BOILERPLATE = [
            r'non[- ]?reportable', r'in the (supreme|high) court',
            r'civil appellate jurisdiction', r'civil appeal no\.',
            r'special leave petition', r'^\s*o\s*r\s*d\s*e\s*r\s*$',
            r'leave granted', r'appellant\(s\)', r'respondent\(s\)',
            r'\.{2,}|…',
        ]
        sentences = [s for s in sentences
                     if not any(re.search(p, s, re.IGNORECASE) for p in BOILERPLATE)]
        scored = []
        total = max(len(sentences), 1)
        for i, sent in enumerate(sentences):
            if len(sent.split()) < 8:
                continue
            score = 0.0
            lower = sent.lower()
            rel = i / total
            if any(w in lower for w in ['held', 'holding', 'court held']): score += 6.0
            if any(w in lower for w in ['allowed', 'dismissed', 'granted', 'rejected',
                                         'set aside', 'upheld', 'quashed', 'acquitted']): score += 4.0
            if any(w in lower for w in ['ruled', 'concluded', 'observed', 'directed',
                                         'noted', 'found']): score += 2.5
            if any(w in lower for w in ['issue', 'whether', 'question', 'contention']): score += 2.0
            if 'court' in lower: score += 0.5
            if any(w in lower for w in ['appeal', 'petition', 'writ']): score += 0.8
            if len(sent.split()) > 25: score += 0.5
            if rel < 0.10: score += 1.0
            if rel > 0.85: score += 3.5
            if score > 0:
                scored.append((score, i, sent))
        if not scored:
            logger.warning("No sentences scored — using raw text truncated")
            words = text.split()
            return ' '.join(words[:350])
        scored.sort(reverse=True, key=lambda x: x[0])
        top = sorted(scored[:self.config.max_extract_sentences], key=lambda x: x[1])
        result = ' '.join(s for _, _, s in top)
        words = result.split()
        if len(words) > 350:
            truncated = ' '.join(words[:350])
            pos = truncated.rfind('.')
            if pos > len(truncated) * 0.6:
                result = truncated[:pos+1]
            else:
                result = truncated + '.'
        logger.info(f"Extraction: {len(result.split())} words -> model")
        return result


# ---------------------------------------------------------------------------
# Weight loading utility
# ---------------------------------------------------------------------------

def load_weights(model, weights_path: str, device: torch.device) -> None:
    """
    Robustly load fine-tuned weights from a .pt / .bin file or a
    HuggingFace save_pretrained() directory.

    Handles all common checkpoint formats:
      • Raw state dict:            {param_name: tensor, ...}
      • Full training checkpoint:  {'model_state_dict': ..., 'epoch': ..., ...}
      • 'state_dict' wrapper:      {'state_dict': ...}
      • DataParallel prefix:       keys starting with 'model.'
    """
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Weights path not found: {weights_path}")

    # ── HuggingFace save_pretrained directory ─────────────────────────────
    # The model was already loaded via from_pretrained(directory), so this
    # path is only reached if the caller explicitly passes a .pt/.bin file.
    if path.is_dir():
        logger.info(f"Directory passed as weights path — already loaded via from_pretrained: {path}")
        return

    # ── Load the file ─────────────────────────────────────────────────────
    logger.info(f"Loading weights from: {weights_path}")
    # weights_only=False needed for checkpoints that stored non-tensor objects
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    # ── Unwrap checkpoint envelope ────────────────────────────────────────
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            # Full training checkpoint saved with:
            # torch.save({'model_state_dict': model.state_dict(), 'epoch': ...})
            sd = checkpoint['model_state_dict']
            meta = {k: v for k, v in checkpoint.items()
                    if k not in ('model_state_dict', 'optimizer_state_dict',
                                 'scheduler_state_dict')}
            if meta:
                logger.info(f"Checkpoint metadata: {meta}")
        elif 'state_dict' in checkpoint:
            sd = checkpoint['state_dict']
        else:
            # Assume it's already a raw state dict
            sd = checkpoint
    else:
        # Some people do torch.save(model.state_dict()) — returns an OrderedDict
        sd = checkpoint

    # ── Strip DataParallel / module prefix ────────────────────────────────
    # When trained with nn.DataParallel, every key is prefixed with 'module.'
    # When wrapped in a custom class, sometimes prefixed with 'model.'
    first_key = next(iter(sd.keys()), '')
    if first_key.startswith('module.'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
        logger.info("Stripped 'module.' prefix (DataParallel checkpoint)")
    elif first_key.startswith('model.'):
        sd = {k[len('model.'):]: v for k, v in sd.items()}
        logger.info("Stripped 'model.' prefix from state dict keys")

    # ── Load with partial-match tolerance ────────────────────────────────
    # strict=False allows loading even if a few keys differ (e.g. added heads).
    # We log exactly what matched and what didn't so you can debug.
    result = model.load_state_dict(sd, strict=False)

    n_loaded    = len(sd) - len(result.missing_keys) - len(result.unexpected_keys)
    n_total     = len(list(model.state_dict().keys()))

    if result.missing_keys:
        logger.warning(
            f"Missing keys ({len(result.missing_keys)}) — "
            f"these params keep their pretrained values:\n"
            + '\n'.join(f"  {k}" for k in result.missing_keys[:8])
            + (f"\n  ... and {len(result.missing_keys)-8} more"
               if len(result.missing_keys) > 8 else "")
        )
    if result.unexpected_keys:
        logger.warning(
            f"Unexpected keys ({len(result.unexpected_keys)}) — "
            f"these were in your checkpoint but not in the model (ignored):\n"
            + '\n'.join(f"  {k}" for k in result.unexpected_keys[:8])
            + (f"\n  ... and {len(result.unexpected_keys)-8} more"
               if len(result.unexpected_keys) > 8 else "")
        )
    if not result.missing_keys and not result.unexpected_keys:
        logger.info(f"✓ All {n_total} weights loaded perfectly from fine-tuned checkpoint")
    else:
        logger.info(f"✓ Loaded {n_loaded}/{n_total} weights from fine-tuned checkpoint")


# ---------------------------------------------------------------------------
# ModelWrapper — now supports pretrained-only vs fine-tuned modes cleanly
# ---------------------------------------------------------------------------

class ModelWrapper:
    """
    Two loading modes, selected by whether a weights_path is supplied:

    Mode A — pretrained only (weights_path=None):
        Loads the base HuggingFace model as-is.  Good for quick testing or
        when you have no fine-tuned checkpoint yet.

    Mode B — fine-tuned (weights_path=<path to .pt/.bin or HF directory>):
        Loads the base architecture first (so the tokenizer and config match),
        then overlays your fine-tuned weights via load_weights().
        This is the correct pattern for HuggingFace seq2seq models — you
        never want to skip the from_pretrained() step entirely because the
        tokenizer vocab and model config must still be read from the hub.
    """

    def __init__(self, model_id: str, device_config: Dict[str, Any],
                 config: SummaryConfig, weights_path: Optional[str] = None):
        self.config  = config
        self.device  = device_config['device']
        self.dtype   = device_config['dtype']
        self.model_id = model_id

        # ── Always load tokenizer from HuggingFace ────────────────────────
        logger.info(f"Loading tokenizer for {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

        # ── Load model architecture ───────────────────────────────────────
        if weights_path and Path(weights_path).is_dir():
            # Fine-tuned checkpoint saved with model.save_pretrained(dir)
            # Load architecture + weights together directly from the directory
            logger.info(f"Loading fine-tuned model from directory: {weights_path}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                weights_path,
                low_cpu_mem_usage=device_config.get('low_cpu_mem', False),
                torch_dtype=self.dtype if self.device.type == 'cuda' else torch.float32,
                ignore_mismatched_sizes=True,
            )
            logger.info(f"✓ Fine-tuned model loaded from {weights_path}")
        else:
            # Load base pretrained architecture from HuggingFace hub
            logger.info(f"Loading base pretrained model: {model_id}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id,
                low_cpu_mem_usage=device_config.get('low_cpu_mem', False),
                torch_dtype=self.dtype if self.device.type == 'cuda' else torch.float32,
            )

            if weights_path:
                # Fine-tuned .pt / .bin file — overlay weights onto base architecture
                logger.info(f"Overlaying fine-tuned weights from: {weights_path}")
                load_weights(self.model, weights_path, self.device)
            else:
                logger.info(f"Using pretrained weights only (no fine-tuned checkpoint supplied)")

        # ── Move to device & optimise ─────────────────────────────────────
        self.model.to(self.device).eval()

        if self.device.type == 'cpu':
            logger.info("Applying dynamic int8 quantization for CPU inference...")
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )

        weight_src = (
            f"fine-tuned ({weights_path})" if weights_path
            else "pretrained (HuggingFace hub)"
        )
        logger.info(f"✓ {model_id} ready  |  weights: {weight_src}  |  device: {self.device}")

    def generate(self, text: str, **kwargs) -> str:
        raise NotImplementedError


class LEDSummarizer(ModelWrapper):
    MODEL_PATH = "nsi319/legal-led-base-16384"

    def __init__(self, device_config: Dict[str, Any], config: SummaryConfig,
                 weights_path: Optional[str] = None):
        super().__init__(self.MODEL_PATH, device_config, config, weights_path)

    def generate(self, text: str, **kwargs) -> str:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.config.max_input_length_led, padding=True
        ).to(self.device)
        gam = torch.zeros_like(inputs.input_ids)
        gam[:, 0] = 1
        with torch.no_grad():
            ids = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                global_attention_mask=gam,
                num_beams=kwargs.get('num_beams', self.config.num_beams),
                min_length=self.config.min_summary_length,
                max_length=self.config.max_summary_length,
                length_penalty=self.config.length_penalty,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                early_stopping=True,
                do_sample=False,
            )
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)


class PegasusSummarizer(ModelWrapper):
    MODEL_PATH = "nsi319/legal-pegasus"

    def __init__(self, device_config: Dict[str, Any], config: SummaryConfig,
                 weights_path: Optional[str] = None):
        super().__init__(self.MODEL_PATH, device_config, config, weights_path)

    def generate(self, text: str, **kwargs) -> str:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.config.max_input_length_pegasus, padding=True
        ).to(self.device)
        with torch.no_grad():
            ids = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                num_beams=kwargs.get('num_beams', self.config.num_beams),
                min_length=self.config.min_summary_length,
                max_length=self.config.max_summary_length,
                length_penalty=self.config.length_penalty,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                early_stopping=True,
                do_sample=False,
            )
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# LegalSummarizer — wires everything together
# ---------------------------------------------------------------------------

class LegalSummarizer:
    """
    Usage examples
    --------------
    # Pretrained only (no fine-tuned weights):
    s = LegalSummarizer('led')
    s = LegalSummarizer('pegasus')
    s = LegalSummarizer('hybrid')

    # Fine-tuned .pt checkpoint:
    s = LegalSummarizer('led',     led_weights='checkpoints/led_finetuned.pt')
    s = LegalSummarizer('pegasus', pegasus_weights='checkpoints/pegasus_finetuned.pt')
    s = LegalSummarizer('hybrid',  led_weights='ckpt/led.pt', pegasus_weights='ckpt/peg.pt')

    # Fine-tuned HuggingFace save_pretrained directory:
    s = LegalSummarizer('led',     led_weights='checkpoints/led_finetuned_dir/')
    """

    def __init__(self, model_type: str,
                 config: Optional[SummaryConfig] = None,
                 # Fine-tuned weight paths (optional — omit to use pretrained only)
                 led_weights: Optional[str] = None,
                 pegasus_weights: Optional[str] = None):
        self.model_type = model_type.lower()
        self.config     = config or SummaryConfig()
        dev             = DeviceManager.get_device_config()
        self.processor  = TextProcessor()
        self.extractor  = SmartLegalExtractor(self.config, self.processor)
        self.checker    = ConsistencyChecker(dev['device'])
        self.corrector  = JurisdictionCorrector()

        if self.model_type == 'led':
            self.model = LEDSummarizer(dev, self.config, led_weights)

        elif self.model_type == 'pegasus':
            self.model = PegasusSummarizer(dev, self.config, pegasus_weights)

        elif self.model_type == 'hybrid':
            # LED first pass → Pegasus second pass
            # Each model independently uses its own fine-tuned weights if provided
            logger.info("Hybrid mode: loading LED + Pegasus")
            self.led     = LEDSummarizer(dev, self.config, led_weights)
            self.pegasus = PegasusSummarizer(dev, self.config, pegasus_weights)

        else:
            raise ValueError(f"Unknown model_type '{model_type}'. Choose: led | pegasus | hybrid")

    # ------------------------------------------------------------------
    # Internal: run one model in the right mode
    # ------------------------------------------------------------------
    def _generate(self, extract_input: str) -> str:
        if self.model_type == 'hybrid':
            intermediate = self.led.generate(extract_input)
            logger.info(f"LED pass: {len(intermediate.split())} words")
            return self.pegasus.generate(intermediate)
        return self.model.generate(extract_input)

    # ------------------------------------------------------------------
    # Public: summarize a document
    # ------------------------------------------------------------------
    def summarize(self, text: str) -> str:
        logger.info(f"\n{'='*50}\nSUMMARIZATION [{self.model_type.upper()}]\n{'='*50}")
        if not text or not text.strip():
            return "Error: No input text provided"

        logger.info(f"Input: {len(text.split())} words")
        extracted = self.extractor.extract_for_model(text)

        if not extracted or len(extracted.split()) < 10:
            logger.warning("Extraction insufficient — extractive fallback")
            return self.extractor.extract_structured_summary(text)

        best_summary, best_score = "", 0.0

        for attempt in range(self.config.max_retries + 1):
            logger.info(f"\n--- Attempt {attempt + 1}/{self.config.max_retries + 1} ---")

            extract_input = extracted
            if attempt > 0:
                words = extracted.split()
                limit = max(150, len(words) - attempt * 60)
                extract_input = ' '.join(words[:limit])
                logger.info(f"Retry: reduced input to {len(extract_input.split())} words")

            try:
                summary = self._generate(extract_input)
                logger.info(f"Generated: {len(summary.split())} words")
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                if attempt < self.config.max_retries:
                    continue
                return self.extractor.extract_structured_summary(text)

            # ── Post-processing ───────────────────────────────────────────
            summary, n_corrections = self.corrector.correct(summary)
            if n_corrections:
                logger.info(f"Jurisdiction corrector: {n_corrections} fix(es) applied")

            summary = self.processor.clean_legal_text(summary)

            words = summary.split()
            if len(words) > self.config.max_summary_length:
                truncated = ' '.join(words[:self.config.max_summary_length])
                pos = truncated.rfind('.')
                summary = truncated[:pos+1] if pos > len(truncated)*0.6 else truncated + '.'

            logger.info(f"After post-processing: {len(summary.split())} words")

            if not summary or len(summary.split()) < 20:
                if attempt < self.config.max_retries:
                    continue
                return self.extractor.extract_structured_summary(text)

            # ── Quality checks ────────────────────────────────────────────
            try:
                hallucinated, h_issues, h_score = HallucinationDetector.detect_hallucination(summary, text)
                consistent, c_issues, c_score   = self.checker.check_consistency(
                    summary, extracted, self.config.consistency_threshold)
            except Exception as e:
                logger.warning(f"Quality check error: {e}")
                if len(summary.split()) >= 30:
                    return summary
                consistent, hallucinated, c_score, h_score = True, False, 0.5, 0.0

            quality_score = c_score
            if quality_score > best_score:
                best_score, best_summary = quality_score, summary

            if not hallucinated and consistent:
                logger.info(f"Quality checks passed (score={quality_score:.2f})")
                return summary

            if hallucinated:
                logger.warning(f"Hallucination after correction: {h_issues[:2]}")
            if not consistent:
                logger.warning(f"Consistency issues: {c_issues[:2]}")

        # ── Fallback decision ─────────────────────────────────────────────
        if self.config.use_extractive_fallback and best_score < 0.30:
            logger.warning("Quality below threshold — extractive fallback")
            return self.extractor.extract_structured_summary(text)

        logger.warning(f"Best-effort result (score={best_score:.2f})")
        return best_summary if best_summary else self.extractor.extract_structured_summary(text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Legal Document Summarizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Weight loading modes
--------------------
Pretrained only (no weights flag):
  python summarizer.py doc.txt --model led

Fine-tuned .pt checkpoint:
  python summarizer.py doc.txt --model led     --led-weights checkpoints/led.pt
  python summarizer.py doc.txt --model pegasus --pegasus-weights checkpoints/pegasus.pt

Fine-tuned HuggingFace save_pretrained directory:
  python summarizer.py doc.txt --model led     --led-weights checkpoints/led_dir/

Hybrid (mix pretrained LED + fine-tuned Pegasus):
  python summarizer.py doc.txt --model hybrid  --pegasus-weights checkpoints/pegasus.pt
        """
    )
    parser.add_argument("input_file", help="Path to input legal document (.txt)")
    parser.add_argument("--model", "-m", required=True,
                        choices=["led", "pegasus", "hybrid"],
                        help="Model architecture to use")

    # Fine-tuned weight paths — optional; omit to use pretrained weights only
    wt = parser.add_argument_group("Fine-tuned weights (optional)")
    wt.add_argument("--led-weights",
                    help="Path to fine-tuned LED weights (.pt file or save_pretrained dir)")
    wt.add_argument("--pegasus-weights",
                    help="Path to fine-tuned Pegasus weights (.pt file or save_pretrained dir)")

    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--no-fallback", action="store_true",
                        help="Disable extractive fallback")
    parser.add_argument("--config", "-c",
                        help="JSON file with SummaryConfig overrides")
    args = parser.parse_args()

    try:
        input_path = Path(args.input_file)
        if not input_path.exists():
            logger.error(f"File not found: {args.input_file}")
            sys.exit(1)

        text = input_path.read_text(encoding='utf-8').strip()
        if not text:
            logger.error("Input file is empty")
            sys.exit(1)

        logger.info(f"Loaded: {len(text.split())} words from {input_path.name}")

        config = SummaryConfig(use_extractive_fallback=not args.no_fallback)
        if args.config:
            overrides = json.loads(Path(args.config).read_text())
            for k, v in overrides.items():
                if hasattr(config, k):
                    setattr(config, k, v)
                else:
                    logger.warning(f"Unknown config key ignored: {k}")

        summarizer = LegalSummarizer(
            model_type      = args.model,
            config          = config,
            led_weights     = args.led_weights,
            pegasus_weights = args.pegasus_weights,
        )

        t0      = time.time()
        summary = summarizer.summarize(text)
        elapsed = time.time() - t0

        src_w = len(text.split())
        sum_w = len(summary.split())

        print(f"\n{'='*70}\nFINAL SUMMARY\n{'='*70}")
        print(summary)
        print(f"{'='*70}")
        print(f"Time: {elapsed:.2f}s | Words: {sum_w} | Compression: {src_w/max(sum_w,1):.1f}x")

        out = args.output or f"{input_path.stem}_{args.model}_summary.txt"
        Path(out).write_text(summary, encoding='utf-8')
        logger.info(f"Saved -> {out}")

    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()