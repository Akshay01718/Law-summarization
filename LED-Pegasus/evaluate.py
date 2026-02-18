import torch
import argparse
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import time
import numpy as np
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Check dependencies
def check_dependencies():
    """Check if all required packages are installed"""
    required = ["rouge_score", "bert_score", "nltk"]
    missing = []
    
    for pkg in required:
        try:
            if pkg == "rouge_score":
                import rouge_score
            elif pkg == "bert_score":
                import bert_score
            else:
                __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        logger.error("=" * 70)
        logger.error("MISSING EVALUATION DEPENDENCIES")
        logger.error("=" * 70)
        logger.error(f"\nThe following packages are required but not installed:")
        for pkg in missing:
            logger.error(f"  ❌ {pkg}")
        logger.error(f"\nPlease install them using:")
        if "rouge_score" in missing:
            logger.error(f"  pip install rouge-score")
        if "bert_score" in missing:
            logger.error(f"  pip install bert-score")
        if "nltk" in missing:
            logger.error(f"  pip install nltk")
        logger.error("=" * 70)
        return False
    
    return True

if not check_dependencies():
    logger.error("\nNote: Install evaluation dependencies separately:")
    logger.error("  pip install rouge-score bert-score nltk")
    sys.exit(1)

from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)


@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics"""
    # ROUGE Scores
    rouge1_precision: float
    rouge1_recall: float
    rouge1_f1: float
    rouge2_precision: float
    rouge2_recall: float
    rouge2_f1: float
    rougeL_precision: float
    rougeL_recall: float
    rougeL_f1: float
    rougeLsum_precision: float
    rougeLsum_recall: float
    rougeLsum_f1: float
    
    # BERTScore
    bert_precision: float
    bert_recall: float
    bert_f1: float
    
    # Length metrics
    summary_length: int
    reference_length: int
    compression_ratio: float
    
    # Timing
    evaluation_time: float
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def print_summary(self):
        """Print formatted summary of metrics"""
        print("\n" + "="*70)
        print("EVALUATION METRICS SUMMARY")
        print("="*70)
        
        print("\n📊 ROUGE SCORES:")
        print("-" * 70)
        print(f"{'Metric':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 70)
        print(f"{'ROUGE-1':<15} {self.rouge1_precision:>11.4f} {self.rouge1_recall:>11.4f} {self.rouge1_f1:>11.4f}")
        print(f"{'ROUGE-2':<15} {self.rouge2_precision:>11.4f} {self.rouge2_recall:>11.4f} {self.rouge2_f1:>11.4f}")
        print(f"{'ROUGE-L':<15} {self.rougeL_precision:>11.4f} {self.rougeL_recall:>11.4f} {self.rougeL_f1:>11.4f}")
        print(f"{'ROUGE-Lsum':<15} {self.rougeLsum_precision:>11.4f} {self.rougeLsum_recall:>11.4f} {self.rougeLsum_f1:>11.4f}")
        
        print("\n🤖 BERTSCORE (Semantic Similarity):")
        print("-" * 70)
        print(f"{'Metric':<15} {'Score':<12}")
        print("-" * 70)
        print(f"{'Precision':<15} {self.bert_precision:>11.4f}")
        print(f"{'Recall':<15} {self.bert_recall:>11.4f}")
        print(f"{'F1-Score':<15} {self.bert_f1:>11.4f}")
        
        print("\n📏 LENGTH METRICS:")
        print("-" * 70)
        print(f"Summary Length:      {self.summary_length:>6} words")
        print(f"Reference Length:    {self.reference_length:>6} words")
        print(f"Compression Ratio:   {self.compression_ratio:>6.2f}x")
        
        print("\n⏱️  PERFORMANCE:")
        print("-" * 70)
        print(f"Evaluation Time:     {self.evaluation_time:>6.2f}s")
        print("="*70)


class SummaryEvaluator:
    """Comprehensive evaluation of summaries"""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize evaluator
        
        Args:
            device: Device for BERTScore ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing evaluator on device: {self.device}")
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
            use_stemmer=True
        )
        logger.info("✓ ROUGE scorer initialized")
    
    def evaluate(self, 
                 generated_summary: str, 
                 reference_summary: str,
                 use_bertscore: bool = True) -> EvaluationMetrics:
        """
        Evaluate generated summary against reference
        
        Args:
            generated_summary: The summary produced by the model
            reference_summary: The ground truth/reference summary
            use_bertscore: Whether to calculate BERTScore (slower but semantic)
        
        Returns:
            EvaluationMetrics object with all metrics
        """
        start_time = time.time()
        
        # Validate inputs
        if not generated_summary or not generated_summary.strip():
            raise ValueError("Generated summary is empty")
        if not reference_summary or not reference_summary.strip():
            raise ValueError("Reference summary is empty")
        
        logger.info("\nCalculating ROUGE scores...")
        rouge_scores = self.rouge_scorer.score(reference_summary, generated_summary)
        
        # Extract ROUGE metrics
        metrics_dict = {
            'rouge1_precision': rouge_scores['rouge1'].precision,
            'rouge1_recall': rouge_scores['rouge1'].recall,
            'rouge1_f1': rouge_scores['rouge1'].fmeasure,
            'rouge2_precision': rouge_scores['rouge2'].precision,
            'rouge2_recall': rouge_scores['rouge2'].recall,
            'rouge2_f1': rouge_scores['rouge2'].fmeasure,
            'rougeL_precision': rouge_scores['rougeL'].precision,
            'rougeL_recall': rouge_scores['rougeL'].recall,
            'rougeL_f1': rouge_scores['rougeL'].fmeasure,
            'rougeLsum_precision': rouge_scores['rougeLsum'].precision,
            'rougeLsum_recall': rouge_scores['rougeLsum'].recall,
            'rougeLsum_f1': rouge_scores['rougeLsum'].fmeasure,
        }
        
        # Calculate BERTScore
        if use_bertscore:
            logger.info("Calculating BERTScore (this may take a moment)...")
            try:
                P, R, F1 = bert_score(
                    [generated_summary], 
                    [reference_summary],
                    lang='en',
                    device=self.device,
                    verbose=False
                )
                metrics_dict['bert_precision'] = P.mean().item()
                metrics_dict['bert_recall'] = R.mean().item()
                metrics_dict['bert_f1'] = F1.mean().item()
            except Exception as e:
                logger.warning(f"BERTScore calculation failed: {e}")
                metrics_dict['bert_precision'] = 0.0
                metrics_dict['bert_recall'] = 0.0
                metrics_dict['bert_f1'] = 0.0
        else:
            metrics_dict['bert_precision'] = 0.0
            metrics_dict['bert_recall'] = 0.0
            metrics_dict['bert_f1'] = 0.0
        
        # Length metrics
        gen_words = len(generated_summary.split())
        ref_words = len(reference_summary.split())
        
        metrics_dict['summary_length'] = gen_words
        metrics_dict['reference_length'] = ref_words
        metrics_dict['compression_ratio'] = ref_words / max(gen_words, 1)
        
        # Timing
        metrics_dict['evaluation_time'] = time.time() - start_time
        
        return EvaluationMetrics(**metrics_dict)
    
    def evaluate_batch(self,
                      generated_summaries: List[str],
                      reference_summaries: List[str],
                      use_bertscore: bool = True) -> Tuple[List[EvaluationMetrics], Dict]:
        """
        Evaluate multiple summaries and compute aggregate statistics
        
        Args:
            generated_summaries: List of generated summaries
            reference_summaries: List of reference summaries
            use_bertscore: Whether to calculate BERTScore
        
        Returns:
            Tuple of (list of individual metrics, aggregate statistics)
        """
        if len(generated_summaries) != len(reference_summaries):
            raise ValueError("Number of generated and reference summaries must match")
        
        logger.info(f"\nEvaluating {len(generated_summaries)} summaries...")
        
        all_metrics = []
        for i, (gen, ref) in enumerate(zip(generated_summaries, reference_summaries)):
            logger.info(f"\nEvaluating summary {i+1}/{len(generated_summaries)}...")
            metrics = self.evaluate(gen, ref, use_bertscore)
            all_metrics.append(metrics)
        
        # Calculate aggregate statistics
        aggregate = self._compute_aggregate_stats(all_metrics)
        
        return all_metrics, aggregate
    
    def _compute_aggregate_stats(self, metrics_list: List[EvaluationMetrics]) -> Dict:
        """Compute mean and std for all metrics"""
        aggregate = defaultdict(lambda: {'mean': 0.0, 'std': 0.0})
        
        # Get all numeric fields
        sample_dict = metrics_list[0].to_dict()
        
        for key in sample_dict.keys():
            if isinstance(sample_dict[key], (int, float)):
                values = [m.to_dict()[key] for m in metrics_list]
                aggregate[key]['mean'] = np.mean(values)
                aggregate[key]['std'] = np.std(values)
        
        return dict(aggregate)


def save_results(metrics: EvaluationMetrics, 
                output_path: Path,
                generated_summary: str,
                reference_summary: str):
    """Save evaluation results to JSON and text files"""
    
    # Save metrics as JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    logger.info(f"✓ Metrics saved to: {json_path}")
    
    # Save detailed report as text
    txt_path = output_path.with_suffix('.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("GENERATED SUMMARY:\n")
        f.write("-"*70 + "\n")
        f.write(generated_summary + "\n\n")
        
        f.write("REFERENCE SUMMARY:\n")
        f.write("-"*70 + "\n")
        f.write(reference_summary + "\n\n")
        
        f.write("METRICS:\n")
        f.write("-"*70 + "\n")
        for key, value in metrics.to_dict().items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"✓ Report saved to: {txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Legal Document Summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single summary
  python evaluate_summarizer.py generated.txt reference.txt
  
  # Evaluate without BERTScore (faster)
  python evaluate_summarizer.py generated.txt reference.txt --no-bertscore
  
  # Batch evaluation from directory
  python evaluate_summarizer.py --batch summaries_dir/ references_dir/
        """
    )
    
    parser.add_argument("generated", nargs='?', help="Generated summary file or directory")
    parser.add_argument("reference", nargs='?', help="Reference summary file or directory")
    parser.add_argument("--batch", action="store_true", 
                       help="Batch mode: evaluate all files in directories")
    parser.add_argument("--no-bertscore", action="store_true",
                       help="Skip BERTScore calculation (faster)")
    parser.add_argument("--output", "-o", help="Output file for results")
    parser.add_argument("--device", choices=['cuda', 'cpu'], 
                       help="Device for BERTScore (default: auto)")
    
    args = parser.parse_args()
    
    if not args.generated or not args.reference:
        parser.print_help()
        sys.exit(1)
    
    try:
        evaluator = SummaryEvaluator(device=args.device)
        
        if args.batch:
            # Batch evaluation
            gen_dir = Path(args.generated)
            ref_dir = Path(args.reference)
            
            if not gen_dir.is_dir() or not ref_dir.is_dir():
                logger.error("In batch mode, both arguments must be directories")
                sys.exit(1)
            
            # Find matching files
            gen_files = sorted(gen_dir.glob("*.txt"))
            ref_files = sorted(ref_dir.glob("*.txt"))
            
            if len(gen_files) != len(ref_files):
                logger.warning(f"Mismatch: {len(gen_files)} generated, {len(ref_files)} reference files")
            
            generated_summaries = []
            reference_summaries = []
            
            for gen_file, ref_file in zip(gen_files, ref_files):
                with open(gen_file, 'r', encoding='utf-8') as f:
                    generated_summaries.append(f.read().strip())
                with open(ref_file, 'r', encoding='utf-8') as f:
                    reference_summaries.append(f.read().strip())
            
            # Evaluate batch
            all_metrics, aggregate = evaluator.evaluate_batch(
                generated_summaries,
                reference_summaries,
                use_bertscore=not args.no_bertscore
            )
            
            # Print aggregate results
            print("\n" + "="*70)
            print("AGGREGATE STATISTICS")
            print("="*70)
            print(f"\nEvaluated {len(all_metrics)} summaries\n")
            print(f"{'Metric':<25} {'Mean':<12} {'Std Dev':<12}")
            print("-" * 70)
            for key, stats in aggregate.items():
                print(f"{key:<25} {stats['mean']:>11.4f} {stats['std']:>11.4f}")
            print("="*70)
            
            # Save aggregate results
            output_path = Path(args.output) if args.output else Path("batch_evaluation_results.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'individual_metrics': [m.to_dict() for m in all_metrics],
                    'aggregate_statistics': aggregate
                }, f, indent=2)
            logger.info(f"\n✓ Results saved to: {output_path}")
            
        else:
            # Single file evaluation
            gen_path = Path(args.generated)
            ref_path = Path(args.reference)
            
            if not gen_path.exists():
                logger.error(f"Generated summary file not found: {gen_path}")
                sys.exit(1)
            if not ref_path.exists():
                logger.error(f"Reference summary file not found: {ref_path}")
                sys.exit(1)
            
            with open(gen_path, 'r', encoding='utf-8') as f:
                generated = f.read().strip()
            with open(ref_path, 'r', encoding='utf-8') as f:
                reference = f.read().strip()
            
            # Evaluate
            metrics = evaluator.evaluate(
                generated, 
                reference,
                use_bertscore=not args.no_bertscore
            )
            
            # Print results
            metrics.print_summary()
            
            # Save results
            output_path = Path(args.output) if args.output else Path("evaluation_results")
            save_results(metrics, output_path, generated, reference)
    
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()