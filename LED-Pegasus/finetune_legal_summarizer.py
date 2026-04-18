"""
Fine-tuning script for Legal Summarizer (LED + Pegasus + Flan-T5) on IN-ABS dataset

Usage:
    python finetune_legal_summarizer.py --model pegasus --epochs 3
    python finetune_legal_summarizer.py --model pegasus --epochs 3 --subset 500   # fast run
    python finetune_legal_summarizer.py --model hybrid  --epochs 3 --subset 1000 --fp16
    python finetune_legal_summarizer.py --model all     --epochs 3 --subset 1000 --fp16
"""

import os
import sys
import logging
import argparse
import warnings
import time
from dataclasses import dataclass

import torch
import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ── Dependency check ──────────────────────────────────────────────────────────
def check_dependencies():
    required = ["torch", "transformers", "datasets", "evaluate", "rouge_score", "sentencepiece"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error(f"Missing: {missing}")
        logger.error(f"Install: pip install {' '.join(missing)}")
        sys.exit(1)

check_dependencies()

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import load_dataset
import evaluate


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class TrainingConfig:
    dataset_name: str = "percins/IN-ABS"

    # Model IDs
    led_model_id: str     = "nsi319/legal-led-base-16384"
    pegasus_model_id: str = "nsi319/legal-pegasus"
    flan_t5_model_id: str = "google/flan-t5-base"

    # ── Reduced input lengths for speed ──────────────────────────────────────
    # Pegasus: 256 instead of 512  → ~2x faster tokenisation & attention
    # LED:    1024 instead of 4096 → much faster on CPU fallback too
    max_input_length_led: int     = 1024
    max_input_length_pegasus: int = 256   # KEY speed-up for Pegasus
    max_input_length_flan_t5: int = 256
    max_target_length: int        = 128   # 128 is enough for legal summaries

    # Training hyperparams
    epochs: int           = 3
    batch_size: int       = 4     # higher batch = faster on GPU
    grad_accum_steps: int = 4     # effective batch = 4*4 = 16
    learning_rate: float  = 2e-5
    warmup_ratio: float   = 0.05
    weight_decay: float   = 0.01

    fp16: bool = True    # default ON for CUDA — biggest single speed-up

    # ── Subset settings ───────────────────────────────────────────────────────
    # None = use full dataset | int = use only N samples
    train_subset: int = 1000   # ~1000 samples trains in minutes on GPU
    eval_subset:  int = 200    # keep eval small too

    output_dir: str = "./legal_summarizer_finetuned"


# ── Column Discovery ──────────────────────────────────────────────────────────
def discover_columns(dataset):
    cols = dataset["train"].column_names
    logger.info(f"Dataset columns: {cols}")
    input_col  = next((c for c in ["judgement", "judgment", "text", "document"] if c in cols), cols[0])
    target_col = next((c for c in ["summary", "abstract"] if c in cols), cols[1])
    logger.info(f"Using -> input: '{input_col}' | target: '{target_col}'")
    return input_col, target_col


# ── Preprocessing ─────────────────────────────────────────────────────────────
def build_preprocess_fn(tokenizer, input_col, target_col, max_input, max_target, is_led):

    def preprocess(examples):
        inputs = [
            "Summarize the legal case by stating the key issues, procedural history, and outcome: " + str(t)
            for t in examples[input_col]
        ]
        targets = [str(t) for t in examples[target_col]]

        model_inputs = tokenizer(
            inputs,
            max_length=max_input,
            truncation=True,
            padding="max_length",
        )

        labels = tokenizer(
            text_target=targets,
            max_length=max_target,
            truncation=True,
            padding="max_length",
        )

        # -100 tells the loss to ignore padding tokens
        model_inputs["labels"] = [
            [(tok if tok != tokenizer.pad_token_id else -100) for tok in label]
            for label in labels["input_ids"]
        ]

        # LED needs global attention on the first token
        if is_led:
            gam = [[0] * len(ids) for ids in model_inputs["input_ids"]]
            for row in gam:
                row[0] = 1
            model_inputs["global_attention_mask"] = gam

        return model_inputs

    return preprocess


# ── Metrics ───────────────────────────────────────────────────────────────────
def build_compute_metrics(tokenizer):
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(
            predictions=[p.strip() for p in decoded_preds],
            references=[l.strip() for l in decoded_labels],
            use_stemmer=True,
        )
        return {k: round(v * 100, 4) for k, v in result.items()}

    return compute_metrics


# ── Core Fine-tune ────────────────────────────────────────────────────────────
def finetune(model_id: str, cfg: TrainingConfig, max_input: int,
             is_led: bool, suffix: str):

    out_dir = os.path.join(cfg.output_dir, suffix)
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Fine-tuning  : {model_id}")
    logger.info(f"Epochs       : {cfg.epochs}")
    logger.info(f"Max input len: {max_input} tokens")
    logger.info(f"Train subset : {cfg.train_subset} samples")
    logger.info(f"Eval  subset : {cfg.eval_subset} samples")
    logger.info(f"FP16         : {cfg.fp16}")
    logger.info(f"Output dir   : {out_dir}")
    logger.info(f"{'='*60}")

    # ── Load & subset dataset ─────────────────────────────────────────────────
    logger.info("Loading dataset ...")
    dataset   = load_dataset(cfg.dataset_name)
    input_col, target_col = discover_columns(dataset)

    train_ds = dataset["train"]
    eval_ds  = dataset.get("validation") or dataset.get("test")

    # Apply subset BEFORE tokenising — much faster map()
    if cfg.train_subset and cfg.train_subset < len(train_ds):
        train_ds = train_ds.select(range(cfg.train_subset))
        logger.info(f"Train subset : using {cfg.train_subset} / {len(dataset['train'])} samples")

    if eval_ds and cfg.eval_subset and cfg.eval_subset < len(eval_ds):
        eval_ds = eval_ds.select(range(cfg.eval_subset))
        logger.info(f"Eval  subset : using {cfg.eval_subset} samples")

    # ── Tokenizer & Model ─────────────────────────────────────────────────────
    logger.info("Loading tokenizer & model ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU   : {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    model.to(device)

    # ── Tokenise (only on the subset — fast) ──────────────────────────────────
    logger.info("Tokenising ...")
    preprocess_fn = build_preprocess_fn(
        tokenizer, input_col, target_col,
        max_input, cfg.max_target_length, is_led,
    )

    tok_train = train_ds.map(preprocess_fn, batched=True, desc="Tokenising train")
    tok_eval  = eval_ds.map(preprocess_fn,  batched=True, desc="Tokenising eval") if eval_ds else None

    logger.info(f"Tokenised -> train: {len(tok_train)} | eval: {len(tok_eval) if tok_eval else 0}")

    # ── Training arguments ────────────────────────────────────────────────────
    # generation_num_beams=1 (greedy) during eval — beam search is slow
    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        num_train_epochs=cfg.epochs,

        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,

        learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=1.0,

        # FP16 — biggest speed boost on CUDA
        fp16=cfg.fp16 and torch.cuda.is_available(),
        bf16=False,

        evaluation_strategy="epoch" if tok_eval else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True if tok_eval else False,
        metric_for_best_model="rouge2" if tok_eval else None,
        greater_is_better=True,

        predict_with_generate=True,
        generation_max_length=cfg.max_target_length,
        generation_num_beams=1,        # greedy decode during eval = much faster

        logging_steps=25,
        report_to="none",

        dataloader_num_workers=2,      # parallel data loading on GPU
        dataloader_pin_memory=True,    # faster CPU→GPU transfers
    )

    callbacks = []
    if tok_eval:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tok_train,
        eval_dataset=tok_eval,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
        compute_metrics=build_compute_metrics(tokenizer) if tok_eval else None,
        callbacks=callbacks if callbacks else None,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    logger.info(f"Training done in {elapsed/60:.1f} min  ({elapsed:.0f}s)")

    # ── Save ──────────────────────────────────────────────────────────────────
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    pt_path = os.path.join(out_dir, f"{suffix}_weights.pt")
    torch.save(model.state_dict(), pt_path)

    logger.info(f"Model saved  -> {out_dir}")
    logger.info(f"Weights (.pt)-> {pt_path}")

    return out_dir, pt_path


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fine-tune Legal Summarizer (fast subset mode)")
    parser.add_argument(
        "--model", "-m", required=True,
        choices=["led", "pegasus", "hybrid", "flan_t5", "all"],
        help="led | pegasus | hybrid (LED+Pegasus) | flan_t5 | all"
    )
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--subset",      type=int,   default=1000,
                        help="Number of training samples to use (default: 1000). Use 0 for full dataset.")
    parser.add_argument("--eval-subset", type=int,   default=200,
                        help="Number of eval samples (default: 200)")
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--grad-accum",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--fp16",        action="store_true", default=True,
                        help="Use FP16 on CUDA (default: ON)")
    parser.add_argument("--no-fp16",     action="store_true",
                        help="Disable FP16")
    parser.add_argument("--output-dir",  type=str,   default="./legal_summarizer_finetuned")

    args = parser.parse_args()

    cfg = TrainingConfig(
        epochs=args.epochs,
        train_subset=args.subset if args.subset > 0 else None,
        eval_subset=args.eval_subset,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=False if args.no_fp16 else args.fp16,
        output_dir=args.output_dir,
    )

    # Print estimated time
    logger.info(f"\n{'='*60}")
    logger.info("SPEED OPTIMISATIONS ACTIVE:")
    logger.info(f"  Subset         : {cfg.train_subset} train / {cfg.eval_subset} eval samples")
    logger.info(f"  FP16           : {cfg.fp16 and torch.cuda.is_available()}")
    logger.info(f"  Pegasus input  : {cfg.max_input_length_pegasus} tokens (vs 512 default)")
    logger.info(f"  Eval beams     : 1 (greedy — no beam search overhead)")
    logger.info(f"  Effective batch: {cfg.batch_size * cfg.grad_accum_steps}")
    logger.info(f"{'='*60}\n")

    results = {}

    if args.model in ("led", "hybrid", "all"):
        led_dir, led_pt = finetune(cfg.led_model_id, cfg, cfg.max_input_length_led, True,  "led")
        results["led_pt"] = led_pt

    if args.model in ("pegasus", "hybrid", "all"):
        peg_dir, peg_pt = finetune(cfg.pegasus_model_id, cfg, cfg.max_input_length_pegasus, False, "pegasus")
        results["pegasus_pt"] = peg_pt

    if args.model in ("flan_t5", "all"):
        ft5_dir, ft5_pt = finetune(cfg.flan_t5_model_id, cfg, cfg.max_input_length_flan_t5, False, "flan_t5")
        results["flan_t5_pt"] = ft5_pt

    # ── Inference usage ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("DONE — use fine-tuned weights with your inference script:")
    print(f"{'='*70}")

    if "led_pt" in results and "pegasus_pt" in results:
        print(f"\npython summarizer.py input.txt --model hybrid \\")
        print(f"    --led-model-path     {results['led_pt']} \\")
        print(f"    --pegasus-model-path {results['pegasus_pt']}")
    elif "led_pt" in results:
        print(f"\npython summarizer.py input.txt --model led --led-model-path {results['led_pt']}")
    elif "pegasus_pt" in results:
        print(f"\npython summarizer.py input.txt --model pegasus --pegasus-model-path {results['pegasus_pt']}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()