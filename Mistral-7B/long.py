#!/usr/bin/env python3
"""
Legal Document Summarizer v5.1
- Fixed: deprecated torch_dtype -> dtype
- Fixed: temperature/do_sample conflict
- Fixed: meta device offloading with explicit memory management
- Single model (Mistral-7B-Instruct), single pass, clean output
"""

import argparse, json, logging, re, sys, time, warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

warnings.filterwarnings("ignore")


def check_dependencies() -> bool:
    required = {"torch": "torch", "transformers": "transformers>=4.35.0",
                "sentencepiece": "sentencepiece", "accelerate": "accelerate"}
    missing = [i for p, i in required.items() if not _can_import(p)]
    if missing:
        print("Missing:", ", ".join(missing))
        print(f"Install: pip install {' '.join(missing)}")
        return False
    return True

def _can_import(name):
    try: __import__(name); return True
    except ImportError: return False

if not check_dependencies():
    sys.exit(1)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

SECTION_HEADINGS = [
    "Court and Citation",
    "Parties",
    "Facts of the Case",
    "Legal Issue(s)",
    "Arguments",
    "Court's Analysis and Reasoning",
    "Decision / Holding",
    "Key Legal Principles Established",
]


@dataclass
class SummarizerConfig:
    model_name: str = DEFAULT_MODEL
    max_input_chars: int = 12_000
    max_new_tokens: int = 1_200
    # FIX: do_sample=False means greedy decoding — temperature must not be passed
    do_sample: bool = False
    use_half_precision: bool = True
    device_map: str = "auto"

    @classmethod
    def from_file(cls, path):
        with open(path) as f: return cls(**json.load(f))
    def save(self, path):
        with open(path, "w") as f: json.dump(asdict(self), f, indent=2)


class TextCleaner:
    _PAGE = re.compile(r"(?:^|\n)\s*-?\s*\d{1,4}\s*-?\s*(?:\n|$)")
    _WS   = re.compile(r"[ \t]{2,}")
    _CITE = re.compile(r"\[\d+\]")

    @classmethod
    def clean(cls, text):
        if not text or not text.strip():
            raise ValueError("Input text is empty.")
        text = cls._PAGE.sub("\n", text)
        text = cls._CITE.sub("", text)
        return cls._WS.sub(" ", text).strip()

    @classmethod
    def truncate(cls, text, max_chars):
        if len(text) <= max_chars:
            return text, False
        cut = text[:max_chars]
        p = cut.rfind(". ")
        if p > max_chars * 0.8:
            cut = cut[:p + 1]
        logger.warning(f"Input truncated to {len(cut):,} chars. Use --max-chars to raise.")
        return cut, True


SYSTEM_PROMPT = """You are a senior legal analyst specialising in Supreme Court judgments.
Produce a structured legal case summary. Rules:
- Formal, precise legal English.
- Do NOT copy large verbatim passages.
- Be concise but analytically complete.
- Explain WHY the court ruled as it did.
- Highlight precedents relied upon.
- If a section cannot be determined, write: Not determinable from the provided text.
- Return ONLY the structured summary, no preamble."""

USER_TEMPLATE = """Summarise the following Supreme Court judgment under EXACTLY these eight headings (verbatim):

1. Court and Citation
2. Parties
3. Facts of the Case
4. Legal Issue(s)
5. Arguments
6. Court's Analysis and Reasoning
7. Decision / Holding
8. Key Legal Principles Established

---
{text}
---

Begin the summary now:"""


def build_prompt(text, model_name):
    user = USER_TEMPLATE.format(text=text)
    n = model_name.lower()
    if "mistral" in n or "mixtral" in n:
        return f"[INST] {SYSTEM_PROMPT}\n\n{user} [/INST]"
    if "phi-3" in n or "phi3" in n:
        return f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n"
    if "llama" in n or "tinyllama" in n:
        return f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{user}\n<|assistant|>\n"
    return f"{SYSTEM_PROMPT}\n\n{user}"


class SummaryParser:
    _RE = re.compile(
        r"^\s*(?:\*{1,2})?\s*(?:\d+\.\s*)?("
        + "|".join(re.escape(h) for h in SECTION_HEADINGS)
        + r")\s*(?:\*{1,2})?\s*:?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    @classmethod
    def parse(cls, raw):
        matches = list(cls._RE.finditer(raw))
        if not matches:
            logger.warning("Could not parse sections — returning raw output.")
            return {"raw_output": raw.strip()}
        sections = {}
        for i, m in enumerate(matches):
            heading = cls._norm(m.group(1))
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
            sections[heading] = raw[start:end].strip() or "Not determinable from the provided text."
        return sections

    @classmethod
    def _norm(cls, h):
        hl = h.lower().strip()
        for c in SECTION_HEADINGS:
            if c.lower() in hl or hl in c.lower():
                return c
        return h.strip()


class LegalSummarizer:
    def __init__(self, config):
        self.config = config
        logger.info(f"Loading model: {config.model_name}")
        t0 = time.time()

        # FIX: use `dtype` not `torch_dtype` (deprecated in newer transformers)
        dtype = torch.float16 if config.use_half_precision else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            use_fast=True,
        )

        # FIX: explicit max_memory to prevent silent meta-device offloading failures.
        # Tells accelerate: use GPU first, overflow to CPU RAM, never spill to disk.
        if config.device_map == "auto" and torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            # Reserve 10% of GPU for activations, give rest to weights
            gpu_alloc = int(gpu_mem * 0.88)
            max_memory = {0: gpu_alloc, "cpu": "24GiB"}
            logger.info(f"GPU memory budget: {gpu_alloc / 1e9:.1f} GB")
        else:
            max_memory = None

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            dtype=dtype,                    # FIX: was torch_dtype
            device_map=config.device_map,
            max_memory=max_memory,          # FIX: prevents disk offloading / meta device hangs
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        logger.info(f"Model ready in {time.time() - t0:.1f}s")

        # Log where layers ended up
        if hasattr(self.model, "hf_device_map"):
            devices = set(str(d) for d in self.model.hf_device_map.values())
            logger.info(f"Model layers on: {', '.join(sorted(devices))}")

    def summarize(self, raw_text):
        t0 = time.time()

        text = TextCleaner.clean(raw_text)
        text, truncated = TextCleaner.truncate(text, self.config.max_input_chars)

        prompt = build_prompt(text, self.config.model_name)

        # Move inputs to the device of the first model parameter
        first_device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(first_device)
        n_in = inputs["input_ids"].shape[-1]
        logger.info(f"Input: {n_in} tokens — generating...")

        # FIX: never pass temperature when do_sample=False — transformers now
        # raises a warning and may silently skip generation if you do.
        generate_kwargs = dict(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,                # greedy decoding — deterministic
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        with torch.no_grad():
            out_ids = self.model.generate(**generate_kwargs)

        # Decode only the newly generated tokens (exclude the prompt)
        new_tokens = out_ids[0][n_in:]
        raw_out = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        if not raw_out.strip():
            logger.error("Model returned empty output. Check GPU memory or try --no-fp16.")

        sections = SummaryParser.parse(raw_out)
        elapsed = round(time.time() - t0, 2)
        logger.info(f"Done in {elapsed}s — {len(sections)} sections parsed")

        return {
            "sections": sections,
            "metadata": {
                "model": self.config.model_name,
                "input_tokens": n_in,
                "input_truncated": truncated,
                "sections_parsed": len(sections),
                "processing_time_seconds": elapsed,
                "timestamp": datetime.now().isoformat(),
            },
            "_raw_output": raw_out,
        }


class OutputFormatter:
    @staticmethod
    def structured(result):
        lines = ["=" * 72, "STRUCTURED LEGAL CASE SUMMARY", "=" * 72, ""]
        s = result["sections"]
        if "raw_output" in s:
            lines += ["(Sections could not be parsed — raw output below)", "", s["raw_output"]]
        else:
            for h in SECTION_HEADINGS:
                content = s.get(h, "Not determinable from the provided text.")
                lines += [h.upper(), "-" * len(h), content, ""]
        m = result["metadata"]
        lines += [
            "=" * 72,
            f"Model  : {m['model']}",
            f"Tokens : {m['input_tokens']}  |  Time: {m['processing_time_seconds']}s",
            f"Sections parsed: {m['sections_parsed']}",
        ]
        if m["input_truncated"]:
            lines.append("WARNING: Input truncated — consider raising --max-chars")
        lines.append("=" * 72)
        return "\n".join(lines)

    @staticmethod
    def as_json(result):
        return json.dumps(
            {k: v for k, v in result.items() if k != "_raw_output"},
            indent=2, ensure_ascii=False
        )

    @classmethod
    def format(cls, result, fmt):
        return cls.as_json(result) if fmt == "json" else cls.structured(result)


def parse_args():
    p = argparse.ArgumentParser(description="Legal Summarizer v5.1")
    p.add_argument("input_file", help="Path to judgment .txt file")
    p.add_argument("-o", "--output", help="Save output to this file")
    p.add_argument("--format", choices=["structured", "json"], default="structured")
    p.add_argument("--max-chars", type=int, default=12_000,
                   help="Max input characters (default: 12000)")
    p.add_argument("--max-tokens", type=int, default=1_200,
                   help="Max tokens to generate (default: 1200)")
    p.add_argument("--config", help="Path to JSON config file")
    p.add_argument("--cpu", action="store_true", help="Force CPU (slow but no VRAM needed)")
    p.add_argument("--no-fp16", action="store_true", help="Use float32 (more VRAM, more stable)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg = SummarizerConfig.from_file(args.config) if args.config else SummarizerConfig()
    cfg.max_input_chars = args.max_chars
    cfg.max_new_tokens  = args.max_tokens

    if args.cpu:
        cfg.device_map = "cpu"
        cfg.use_half_precision = False
        logger.info("CPU mode: this will be slow (minutes per document)")
    if args.no_fp16:
        cfg.use_half_precision = False

    path = Path(args.input_file)
    if not path.exists():
        logger.error(f"File not found: {path}")
        sys.exit(1)

    text = path.read_text(encoding="utf-8")
    logger.info(f"Loaded: {path} ({len(text):,} chars)")

    try:
        summarizer = LegalSummarizer(cfg)
        result = summarizer.summarize(text)
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU out of memory. Try --no-fp16 for more stability, or --cpu to run on RAM.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed: {e}")
        if args.verbose:
            import traceback; traceback.print_exc()
        sys.exit(1)

    out = OutputFormatter.format(result, args.format)
    
    # Auto-generate output filename if not specified
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "json" if args.format == "json" else "txt"
        args.output = f"summary_{path.stem}_{timestamp}.{ext}"
    
    # Always save to file
    Path(args.output).write_text(out, encoding="utf-8")
    logger.info(f"✅ Saved: {args.output}")
    
    # Also print to console
    print("\n" + out + "\n")


if __name__ == "__main__":
    main()