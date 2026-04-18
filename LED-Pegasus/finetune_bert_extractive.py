"""
Fine-tuning BERT for Extractive Summarization on IN-ABS dataset

How it works:
  - Each sentence in the judgement is scored 1 (should be in summary) or 0 (skip)
  - BERT learns to classify which sentences are "summary-worthy"
  - At inference: score all sentences → pick top-N → return as extractive summary

Model used: nlpaueb/legal-bert-base-uncased (pre-trained on legal text)

Usage:
    python finetune_bert_extractive.py --epochs 3
    python finetune_bert_extractive.py --epochs 3 --subset 1000 --fp16
    python finetune_bert_extractive.py --epochs 3 --subset 500  --fp16   # fastest
"""

import os
import sys
import re
import time
import logging
import argparse
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import torch
import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ── Dependency check ──────────────────────────────────────────────────────────
def check_dependencies():
    required = ["torch", "transformers", "datasets", "evaluate", "sklearn", "sentencepiece"]
    missing = []
    for pkg in required:
        try:
            __import__("sklearn" if pkg == "sklearn" else pkg)
        except ImportError:
            missing.append("scikit-learn" if pkg == "sklearn" else pkg)
    if missing:
        logger.error(f"Missing: {missing}")
        logger.error(f"Install: pip install {' '.join(missing)}")
        sys.exit(1)

check_dependencies()

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import load_dataset, Dataset
import evaluate
from sklearn.metrics import f1_score, precision_score, recall_score


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class BertConfig:
    dataset_name: str  = "percins/IN-ABS"
    model_id: str      = "nlpaueb/legal-bert-base-uncased"  # legal-domain BERT

    # Tokenisation
    max_length: int    = 128    # per-sentence length — sentences are short so 128 is enough

    # Subset
    train_subset: int  = 1000
    eval_subset:  int  = 200

    # Training
    epochs: int        = 3
    batch_size: int    = 16     # BERT handles large batches easily
    grad_accum: int    = 2
    learning_rate: float = 2e-5
    warmup_ratio: float  = 0.1
    weight_decay: float  = 0.01
    fp16: bool         = True

    output_dir: str    = "./legal_bert_extractive"


# ── Text utilities ────────────────────────────────────────────────────────────
def split_sentences(text: str) -> List[str]:
    """Split legal text into sentences, handling common abbreviations."""
    if not text or not text.strip():
        return []
    for abbr, repl in [
        (r'\bv\.\s',   'v_P_ '), (r'\bNo\.\s',  'No_P_ '),
        (r'\bvs\.\s',  'vs_P_ '), (r'\bDr\.\s',  'Dr_P_ '),
        (r'\bMr\.\s',  'Mr_P_ '), (r'\bMrs\.\s', 'Mrs_P_ '),
        (r'\bSec\.\s', 'Sec_P_ '),
    ]:
        text = re.sub(abbr, repl, text)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', text)
    sentences = [s.replace('_P_', '.').strip() for s in sentences]
    return [s for s in sentences if len(s.split()) >= 5]  # drop very short fragments


def sentence_overlap_score(sentence: str, summary: str) -> float:
    """
    Compute word overlap between a sentence and the reference summary.
    Returns a score between 0.0 and 1.0.
    Used to create binary labels: overlap > threshold → label=1 (include in summary).
    """
    s_words = set(re.findall(r'\b\w+\b', sentence.lower()))
    r_words = set(re.findall(r'\b\w+\b', summary.lower()))
    if not s_words or not r_words:
        return 0.0
    # Remove very common stopwords for better signal
    stopwords = {'the','a','an','is','are','was','were','in','of','to','and',
                 'that','this','it','for','on','with','as','at','by','from'}
    s_words -= stopwords
    r_words -= stopwords
    if not s_words:
        return 0.0
    return len(s_words & r_words) / len(s_words)


# ── Dataset Builder ───────────────────────────────────────────────────────────
def build_sentence_dataset(
    raw_dataset,
    input_col: str,
    target_col: str,
    subset: int = None,
    overlap_threshold: float = 0.25,
) -> Dataset:
    """
    Convert document-level IN-ABS data into sentence-level classification data.

    Each row in the output dataset:
        sentence : str   — one sentence from the judgement
        label    : int   — 1 if sentence should be in summary, 0 otherwise
    """
    if subset and subset < len(raw_dataset):
        raw_dataset = raw_dataset.select(range(subset))

    sentences_list, labels_list = [], []
    pos, neg = 0, 0

    for example in raw_dataset:
        judgement = str(example.get(input_col, ""))
        summary   = str(example.get(target_col, ""))
        if not judgement or not summary:
            continue

        sentences = split_sentences(judgement)
        for sent in sentences:
            score = sentence_overlap_score(sent, summary)
            label = 1 if score >= overlap_threshold else 0
            sentences_list.append(sent)
            labels_list.append(label)
            if label == 1: pos += 1
            else:          neg += 1

    total = pos + neg
    logger.info(f"Built sentence dataset: {total} sentences | "
                f"positive={pos} ({100*pos/max(total,1):.1f}%) | "
                f"negative={neg} ({100*neg/max(total,1):.1f}%)")

    return Dataset.from_dict({"sentence": sentences_list, "label": labels_list})


# ── Tokenisation ──────────────────────────────────────────────────────────────
def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def tokenize(examples):
        return tokenizer(
            examples["sentence"],
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
    return dataset.map(tokenize, batched=True, desc="Tokenising")


# ── Metrics ───────────────────────────────────────────────────────────────────
def build_compute_metrics():
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
        f1  = f1_score(labels, preds, average="binary", zero_division=0)
        pre = precision_score(labels, preds, average="binary", zero_division=0)
        rec = recall_score(labels, preds, average="binary", zero_division=0)
        return {
            "accuracy":  round(acc, 4),
            "f1":        round(f1,  4),
            "precision": round(pre, 4),
            "recall":    round(rec, 4),
        }
    return compute_metrics


# ── Inference helper (use after training) ────────────────────────────────────
class BertExtractiveInferencer:
    """
    Use the fine-tuned BERT model to extract the best sentences from a document.
    Plug this into your LegalSummarizer as a smarter extract_for_model().
    """

    def __init__(self, model_dir: str, device: str = None):
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device).eval()
        logger.info(f"BERT extractor loaded from {model_dir} on {self.device}")

    def extract(self, text: str, top_n: int = 8, max_words: int = 350) -> str:
        """
        Score all sentences in text and return top_n as a single string.
        Capped at max_words to match your existing pipeline.
        """
        sentences = split_sentences(text)
        if not sentences:
            return text

        scored = []
        for sent in sentences:
            inputs = self.tokenizer(
                sent,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding="max_length",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                prob   = torch.softmax(logits, dim=-1)[0][1].item()  # prob of label=1

            scored.append((prob, sent))

        # Sort by score, take top_n, restore original order
        top = sorted(scored, reverse=True)[:top_n]
        top_sents = [s for _, s in sorted(
            [(sentences.index(s), s) for _, s in top if s in sentences]
        )]

        result = " ".join(top_sents)

        # Hard word cap — same as your SmartLegalExtractor
        words = result.split()
        if len(words) > max_words:
            truncated = " ".join(words[:max_words])
            pos = truncated.rfind(".")
            result = truncated[:pos+1] if pos > len(truncated)*0.6 else truncated + "."

        return result


# ── Main fine-tune ────────────────────────────────────────────────────────────
def finetune(cfg: BertConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Model        : {cfg.model_id}")
    logger.info(f"Task         : Sentence classification for extractive summarization")
    logger.info(f"Epochs       : {cfg.epochs}")
    logger.info(f"Train subset : {cfg.train_subset} documents → N sentences")
    logger.info(f"FP16         : {cfg.fp16 and torch.cuda.is_available()}")
    logger.info(f"{'='*60}\n")

    # ── Load raw dataset ──────────────────────────────────────────────────────
    logger.info("Loading IN-ABS dataset ...")
    raw = load_dataset(cfg.dataset_name)

    cols       = raw["train"].column_names
    input_col  = next((c for c in ["judgement","judgment","text","document"] if c in cols), cols[0])
    target_col = next((c for c in ["summary","abstract"]                     if c in cols), cols[1])
    logger.info(f"Columns -> input: '{input_col}' | target: '{target_col}'")

    # ── Build sentence-level datasets ────────────────────────────────────────
    logger.info("Building sentence-level train dataset ...")
    train_sent = build_sentence_dataset(
        raw["train"], input_col, target_col, subset=cfg.train_subset
    )

    eval_raw  = raw.get("validation") or raw.get("test")
    eval_sent = None
    if eval_raw:
        logger.info("Building sentence-level eval dataset ...")
        eval_sent = build_sentence_dataset(
            eval_raw, input_col, target_col, subset=cfg.eval_subset
        )

    # ── Tokenizer & Model ─────────────────────────────────────────────────────
    logger.info("Loading Legal-BERT tokenizer & model ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    model     = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_id,
        num_labels=2,           # binary: 0=skip, 1=include in summary
        ignore_mismatched_sizes=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU   : {torch.cuda.get_device_name(0)}")
    model.to(device)

    # ── Tokenise ──────────────────────────────────────────────────────────────
    logger.info("Tokenising sentences ...")
    train_tok = tokenize_dataset(train_sent, tokenizer, cfg.max_length)
    eval_tok  = tokenize_dataset(eval_sent,  tokenizer, cfg.max_length) if eval_sent else None

    # Keep only columns the model needs
    keep_cols = ["input_ids", "attention_mask", "token_type_ids", "label"]
    train_tok = train_tok.remove_columns(
        [c for c in train_tok.column_names if c not in keep_cols]
    )
    if eval_tok:
        eval_tok = eval_tok.remove_columns(
            [c for c in eval_tok.column_names if c not in keep_cols]
        )
    train_tok.set_format("torch")
    if eval_tok:
        eval_tok.set_format("torch")

    logger.info(f"Tokenised -> train: {len(train_tok)} sentences | "
                f"eval: {len(eval_tok) if eval_tok else 0} sentences")

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,

        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,

        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=1.0,

        fp16=cfg.fp16 and torch.cuda.is_available(),

        evaluation_strategy="epoch" if eval_tok else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True if eval_tok else False,
        metric_for_best_model="f1" if eval_tok else None,
        greater_is_better=True,

        logging_steps=25,
        report_to="none",

        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    callbacks = []
    if eval_tok:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        compute_metrics=build_compute_metrics(),
        callbacks=callbacks if callbacks else None,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    logger.info(f"\nTraining done in {elapsed/60:.1f} min ({elapsed:.0f}s)")

    # ── Save ──────────────────────────────────────────────────────────────────
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    pt_path = os.path.join(cfg.output_dir, "bert_extractive_weights.pt")
    torch.save(model.state_dict(), pt_path)

    logger.info(f"Model saved  -> {cfg.output_dir}")
    logger.info(f"Weights (.pt)-> {pt_path}")

    return cfg.output_dir


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fine-tune Legal-BERT for extractive summarization")
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--subset",      type=int,   default=1000,
                        help="Number of documents for training (default: 1000). 0 = full dataset.")
    parser.add_argument("--eval-subset", type=int,   default=200)
    parser.add_argument("--batch-size",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--fp16",        action="store_true", default=True)
    parser.add_argument("--no-fp16",     action="store_true")
    parser.add_argument("--output-dir",  type=str,   default="./legal_bert_extractive")
    args = parser.parse_args()

    cfg = BertConfig(
        epochs=args.epochs,
        train_subset=args.subset if args.subset > 0 else None,
        eval_subset=args.eval_subset,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=False if args.no_fp16 else args.fp16,
        output_dir=args.output_dir,
    )

    model_dir = finetune(cfg)

    # ── Usage instructions ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("BERT EXTRACTIVE SUMMARIZER READY")
    print(f"{'='*70}")
    print(f"\nModel saved to: {model_dir}")
    print("""
To use in your summarizer pipeline, replace SmartLegalExtractor.extract_for_model()
with BertExtractiveInferencer:

    from finetune_bert_extractive import BertExtractiveInferencer

    extractor = BertExtractiveInferencer("./legal_bert_extractive")
    extracted = extractor.extract(judgement_text, top_n=8, max_words=350)

Then feed 'extracted' into your LED or Pegasus model as usual.
""")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()