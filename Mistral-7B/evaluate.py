"""
evaluate_summarizer.py
======================
Evaluates legal document summaries and writes results directly to a PDF report.
No browser required — running the script produces evaluation_results.pdf.

Usage:
  python evaluate_summarizer.py generated.txt reference.txt
  python evaluate_summarizer.py generated.txt reference.txt --no-bertscore
  python evaluate_summarizer.py --batch summaries_dir/ references_dir/
"""

import torch
import argparse
import sys
import json
import logging
import io
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import time
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def check_dependencies():
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
        logger.error("MISSING EVALUATION DEPENDENCIES")
        for pkg in missing:
            logger.error(f"  {pkg}")
        logger.error("Install: pip install rouge-score bert-score nltk")
        return False
    return True


if not check_dependencies():
    sys.exit(1)

from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# PDF / chart imports
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image, KeepTogether,
)
from reportlab.platypus.flowables import Flowable


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

P_BLUE    = colors.HexColor('#1d4ed8')
P_PURPLE  = colors.HexColor('#6d28d9')
P_TEAL    = colors.HexColor('#0f766e')
P_SKY     = colors.HexColor('#0369a1')
P_GOOD    = colors.HexColor('#15803d')
P_WARN    = colors.HexColor('#b45309')
P_BAD     = colors.HexColor('#b91c1c')
P_TEXT    = colors.HexColor('#18181b')
P_MUTED   = colors.HexColor('#52525b')
P_CAPTION = colors.HexColor('#a1a1aa')
P_BORDER  = colors.HexColor('#dedad3')
P_BORDER2 = colors.HexColor('#c4bfb8')
P_BG      = colors.HexColor('#f5f4f0')
P_WHITE   = colors.white

GOOD_BG = colors.HexColor('#f0fdf4');  GOOD_BR = colors.HexColor('#bbf7d0')
WARN_BG = colors.HexColor('#fffbeb');  WARN_BR = colors.HexColor('#fde68a')
BAD_BG  = colors.HexColor('#fef2f2');  BAD_BR  = colors.HexColor('#fecaca')

MPL_COLORS = ['#1d4ed8', '#6d28d9', '#0369a1', '#0f766e', '#1d4ed8']


# ---------------------------------------------------------------------------
# Quality helpers
# ---------------------------------------------------------------------------

def quality_label(score: float) -> str:
    if score >= 0.80: return "Excellent"
    if score >= 0.60: return "Good"
    if score >= 0.40: return "Fair"
    return "Poor"


def quality_color_rl(score: float):
    """Return (bg, border, text) reportlab colors."""
    if score >= 0.60: return GOOD_BG, GOOD_BR, P_GOOD
    if score >= 0.40: return WARN_BG, WARN_BR, P_WARN
    return BAD_BG, BAD_BR, P_BAD


# ---------------------------------------------------------------------------
# EvaluationMetrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluationMetrics:
    rouge1_precision: float;  rouge1_recall: float;  rouge1_f1: float
    rouge2_precision: float;  rouge2_recall: float;  rouge2_f1: float
    rougeL_precision: float;  rougeL_recall: float;  rougeL_f1: float
    rougeLsum_precision: float; rougeLsum_recall: float; rougeLsum_f1: float
    bert_precision: float;    bert_recall: float;    bert_f1: float
    summary_length: int;      reference_length: int
    compression_ratio: float; evaluation_time: float

    def to_dict(self) -> Dict:
        return asdict(self)

    def print_summary(self):
        d = self.to_dict()
        print("\n" + "=" * 70)
        print("EVALUATION METRICS SUMMARY")
        print("=" * 70)
        print(f"\n{'Metric':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 70)
        for name, key in [
            ("ROUGE-1", "rouge1"), ("ROUGE-2", "rouge2"),
            ("ROUGE-L", "rougeL"), ("ROUGE-Lsum", "rougeLsum"),
        ]:
            print(f"{name:<15} {d[key+'_precision']:>11.4f} "
                  f"{d[key+'_recall']:>11.4f} {d[key+'_f1']:>11.4f}")
        print(f"\nBERTScore  P:{self.bert_precision:.4f}  "
              f"R:{self.bert_recall:.4f}  F1:{self.bert_f1:.4f}")
        print(f"Length     Gen:{self.summary_length}  "
              f"Ref:{self.reference_length}  Ratio:{self.compression_ratio:.2f}x")
        print("=" * 70)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class SummaryEvaluator:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing evaluator on device: {self.device}")
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
        logger.info("ROUGE scorer initialized")

    def evaluate(self, generated_summary: str, reference_summary: str,
                 use_bertscore: bool = True) -> EvaluationMetrics:
        start_time = time.time()
        if not generated_summary or not generated_summary.strip():
            raise ValueError("Generated summary is empty")
        if not reference_summary or not reference_summary.strip():
            raise ValueError("Reference summary is empty")

        logger.info("Calculating ROUGE scores...")
        rs = self.rouge_scorer.score(reference_summary, generated_summary)

        md = {
            'rouge1_precision':    rs['rouge1'].precision,
            'rouge1_recall':       rs['rouge1'].recall,
            'rouge1_f1':           rs['rouge1'].fmeasure,
            'rouge2_precision':    rs['rouge2'].precision,
            'rouge2_recall':       rs['rouge2'].recall,
            'rouge2_f1':           rs['rouge2'].fmeasure,
            'rougeL_precision':    rs['rougeL'].precision,
            'rougeL_recall':       rs['rougeL'].recall,
            'rougeL_f1':           rs['rougeL'].fmeasure,
            'rougeLsum_precision': rs['rougeLsum'].precision,
            'rougeLsum_recall':    rs['rougeLsum'].recall,
            'rougeLsum_f1':        rs['rougeLsum'].fmeasure,
        }

        if use_bertscore:
            logger.info("Calculating BERTScore (may take a moment)...")
            try:
                P, R, F1 = bert_score(
                    [generated_summary], [reference_summary],
                    lang='en', device=self.device, verbose=False,
                )
                md['bert_precision'] = P.mean().item()
                md['bert_recall']    = R.mean().item()
                md['bert_f1']        = F1.mean().item()
            except Exception as e:
                logger.warning(f"BERTScore failed: {e}")
                md['bert_precision'] = md['bert_recall'] = md['bert_f1'] = 0.0
        else:
            md['bert_precision'] = md['bert_recall'] = md['bert_f1'] = 0.0

        gw = len(generated_summary.split())
        rw = len(reference_summary.split())
        md['summary_length']    = gw
        md['reference_length']  = rw
        md['compression_ratio'] = rw / max(gw, 1)
        md['evaluation_time']   = time.time() - start_time
        return EvaluationMetrics(**md)

    def evaluate_batch(self, generated_summaries: List[str],
                       reference_summaries: List[str],
                       use_bertscore: bool = True
                       ) -> Tuple[List[EvaluationMetrics], Dict]:
        if len(generated_summaries) != len(reference_summaries):
            raise ValueError("Mismatched number of summaries")
        logger.info(f"Evaluating {len(generated_summaries)} summaries...")
        all_metrics = []
        for i, (gen, ref) in enumerate(zip(generated_summaries, reference_summaries)):
            logger.info(f"  {i+1}/{len(generated_summaries)}...")
            all_metrics.append(self.evaluate(gen, ref, use_bertscore))
        return all_metrics, self._compute_aggregate_stats(all_metrics)

    def _compute_aggregate_stats(self, metrics_list: List[EvaluationMetrics]) -> Dict:
        aggregate: Dict = {}
        sample = metrics_list[0].to_dict()
        for key, val in sample.items():
            if isinstance(val, (int, float)):
                values = [m.to_dict()[key] for m in metrics_list]
                aggregate[key] = {
                    'mean': float(np.mean(values)),
                    'std':  float(np.std(values)),
                }
        return aggregate


# ---------------------------------------------------------------------------
# Confusion matrix helper
# ---------------------------------------------------------------------------

def build_confusion_matrix(generated: str, reference: str) -> dict:
    gw = set(generated.lower().split())
    rw = set(reference.lower().split())
    tp = len(gw & rw)
    fp = len(gw - rw)
    fn = len(rw - gw)
    tn = max(50, len(gw | rw))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / (tp + tn + fp + fn)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, f1=f1, accuracy=accuracy)


# ---------------------------------------------------------------------------
# Matplotlib chart builders  (return PNG bytes)
# ---------------------------------------------------------------------------

_MPL_STYLE = {
    'font.family':        'DejaVu Sans',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'axes.grid.axis':     'x',
    'grid.color':         '#e4e0d8',
    'grid.linewidth':     0.6,
    'axes.facecolor':     '#ffffff',
    'figure.facecolor':   '#ffffff',
    'xtick.labelsize':    7,
    'ytick.labelsize':    7,
    'legend.fontsize':    7,
    'axes.labelsize':     8,
}


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf.read()


def chart_pr_grouped(d: dict) -> bytes:
    """Grouped bar: Precision vs Recall across ROUGE variants."""
    with plt.rc_context(_MPL_STYLE):
        fig, ax = plt.subplots(figsize=(4.2, 2.4))
        labels  = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE-Lsum']
        prec    = [d['rouge1_precision'], d['rouge2_precision'],
                   d['rougeL_precision'], d['rougeLsum_precision']]
        rec     = [d['rouge1_recall'],    d['rouge2_recall'],
                   d['rougeL_recall'],    d['rougeLsum_recall']]
        x  = np.arange(len(labels))
        w  = 0.35
        b1 = ax.bar(x - w/2, prec, w, label='Precision',
                    color='#1d4ed8', alpha=0.82, zorder=3)
        b2 = ax.bar(x + w/2, rec,  w, label='Recall',
                    color='#0f766e', alpha=0.78, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
        ax.legend(loc='upper right', frameon=False)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#e4e0d8', linewidth=0.6)
        ax.grid(axis='x', visible=False)
        for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
        fig.tight_layout(pad=0.4)
        return _fig_to_bytes(fig)


def chart_f1_horizontal(d: dict) -> bytes:
    """Horizontal bar: F1 scores for all metrics."""
    with plt.rc_context(_MPL_STYLE):
        fig, ax = plt.subplots(figsize=(4.2, 2.4))
        labels = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE-Lsum', 'BERTScore']
        values = [d['rouge1_f1'], d['rouge2_f1'], d['rougeL_f1'],
                  d['rougeLsum_f1'], d['bert_f1']]
        bar_colors = ['#1d4ed8', '#6d28d9', '#0369a1', '#0f766e', '#1d4ed8']
        alphas     = [0.85, 0.82, 0.80, 0.78, 0.55]
        y = np.arange(len(labels))
        for i, (v, c, a) in enumerate(zip(values, bar_colors, alphas)):
            ax.barh(y[i], v, color=c, alpha=a, zorder=3, height=0.55)
            ax.text(v + 0.01, y[i], f'{v:.3f}', va='center',
                    fontsize=6.5, color='#52525b')
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(0, 1.12)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#e4e0d8', linewidth=0.6)
        ax.grid(axis='y', visible=False)
        for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
        fig.tight_layout(pad=0.4)
        return _fig_to_bytes(fig)


def chart_radar(d: dict) -> bytes:
    """Radar / spider chart of all metric dimensions."""
    labels = ['R1-P', 'R1-R', 'R1-F1', 'R2-P', 'R2-R', 'R2-F1',
              'RL-F1', 'BERT-P', 'BERT-R', 'BERT-F1']
    vals = [d['rouge1_precision'], d['rouge1_recall'], d['rouge1_f1'],
            d['rouge2_precision'], d['rouge2_recall'], d['rouge2_f1'],
            d['rougeL_f1'],
            d['bert_precision'], d['bert_recall'], d['bert_f1']]
    N    = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    vals_plot = vals + vals[:1]

    with plt.rc_context(_MPL_STYLE):
        fig, ax = plt.subplots(figsize=(3.2, 3.2), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.plot(angles, vals_plot, linewidth=1.5, color='#1d4ed8')
        ax.fill(angles, vals_plot, alpha=0.12, color='#1d4ed8')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=6.5, color='#52525b')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'], size=5.5, color='#a1a1aa')
        ax.grid(color='#e4e0d8', linewidth=0.5)
        ax.spines['polar'].set_color('#e4e0d8')
        fig.tight_layout(pad=0.3)
        return _fig_to_bytes(fig)


def chart_length_donut(gen_len: int, ref_len: int) -> bytes:
    """Donut chart comparing generated vs reference word counts."""
    with plt.rc_context(_MPL_STYLE):
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        sizes  = [gen_len, ref_len]
        clrs   = ['#1d4ed8', '#0f766e']
        wedges, texts = ax.pie(
            sizes, colors=clrs, startangle=90,
            wedgeprops=dict(width=0.48, edgecolor='white', linewidth=2),
        )
        ax.legend(
            wedges,
            [f'Generated ({gen_len}w)', f'Reference ({ref_len}w)'],
            loc='lower center', bbox_to_anchor=(0.5, -0.14),
            frameon=False, fontsize=7,
        )
        ax.set(aspect='equal')
        fig.tight_layout(pad=0.3)
        return _fig_to_bytes(fig)


def chart_confusion_bubble(cm: dict) -> bytes:
    """Bubble chart representing the confusion matrix quadrants."""
    with plt.rc_context(_MPL_STYLE):
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        positions = {'TP': (0.27, 0.73), 'FN': (0.73, 0.73),
                     'FP': (0.27, 0.27), 'TN': (0.73, 0.27)}
        clrs      = {'TP': '#15803d', 'FN': '#b45309',
                     'FP': '#b91c1c', 'TN': '#1e40af'}
        values    = {'TP': cm['tp'], 'FN': cm['fn'],
                     'FP': cm['fp'], 'TN': cm['tn']}
        max_v = max(values.values()) or 1
        for key, (x, y) in positions.items():
            v    = values[key]
            size = max(400, (v / max_v) * 3200)
            ax.scatter(x, y, s=size, color=clrs[key], alpha=0.7, zorder=3)
            ax.text(x, y, str(v), ha='center', va='center',
                    fontsize=8, fontweight='bold', color='white', zorder=4)
            ax.text(x, y - 0.14, key, ha='center', va='top',
                    fontsize=6.5, color=clrs[key], fontweight='bold')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('off')
        ax.text(0.27, 0.95, 'Predicted +', ha='center', va='top',
                fontsize=6.5, color='#a1a1aa')
        ax.text(0.73, 0.95, 'Predicted −', ha='center', va='top',
                fontsize=6.5, color='#a1a1aa')
        ax.text(0.02, 0.73, 'Actual +', ha='left', va='center',
                fontsize=6.5, color='#a1a1aa', rotation=90)
        ax.text(0.02, 0.27, 'Actual −', ha='left', va='center',
                fontsize=6.5, color='#a1a1aa', rotation=90)
        fig.tight_layout(pad=0.3)
        return _fig_to_bytes(fig)


def chart_trend(all_metrics: List[EvaluationMetrics]) -> bytes:
    """Line chart of F1 scores across summaries (batch mode)."""
    with plt.rc_context(_MPL_STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 2.6))
        x = range(1, len(all_metrics) + 1)
        series = [
            ('ROUGE-1',   [m.rouge1_f1  for m in all_metrics], '#1d4ed8'),
            ('ROUGE-2',   [m.rouge2_f1  for m in all_metrics], '#6d28d9'),
            ('ROUGE-L',   [m.rougeL_f1  for m in all_metrics], '#0369a1'),
            ('BERTScore', [m.bert_f1    for m in all_metrics], '#0f766e'),
        ]
        for label, vals, c in series:
            ax.plot(x, vals, marker='o', markersize=3.5, linewidth=1.4,
                    color=c, label=label, zorder=3)
        ax.set_xlim(0.5, len(all_metrics) + 0.5)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Summary #', fontsize=7)
        ax.set_ylabel('F1 Score', fontsize=7)
        ax.legend(loc='upper right', frameon=False, ncol=2)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#e4e0d8', linewidth=0.6)
        ax.grid(axis='x', visible=False)
        for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
        fig.tight_layout(pad=0.4)
        return _fig_to_bytes(fig)


def chart_mean_std(aggregate: Dict) -> bytes:
    """Bar chart of mean F1 ± std (batch mode)."""
    with plt.rc_context(_MPL_STYLE):
        fig, ax = plt.subplots(figsize=(4.2, 2.4))
        labels  = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'ROUGE-Lsum', 'BERTScore']
        keys    = ['rouge1_f1', 'rouge2_f1', 'rougeL_f1', 'rougeLsum_f1', 'bert_f1']
        means   = [aggregate.get(k, {}).get('mean', 0) for k in keys]
        stds    = [aggregate.get(k, {}).get('std',  0) for k in keys]
        bar_colors = ['#1d4ed8', '#6d28d9', '#0369a1', '#0f766e', '#1d4ed8']
        x = np.arange(len(labels))
        ax.bar(x, means, color=bar_colors, alpha=0.82, zorder=3,
               yerr=stds, capsize=4, error_kw=dict(elinewidth=1, ecolor='#71717a'))
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5, rotation=15, ha='right')
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Mean F1', fontsize=7)
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#e4e0d8', linewidth=0.6)
        ax.grid(axis='x', visible=False)
        for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
        fig.tight_layout(pad=0.4)
        return _fig_to_bytes(fig)


# ---------------------------------------------------------------------------
# ReportLab style sheet
# ---------------------------------------------------------------------------

def _make_styles():
    base = getSampleStyleSheet()

    styles = {}

    styles['eyebrow'] = ParagraphStyle(
        'eyebrow', fontName='Helvetica', fontSize=7,
        textColor=P_CAPTION, spaceBefore=0, spaceAfter=2,
        leading=9, letterSpacing=1.5,
    )
    styles['title'] = ParagraphStyle(
        'title', fontName='Times-Bold', fontSize=22,
        textColor=P_TEXT, spaceBefore=0, spaceAfter=4, leading=26,
    )
    styles['section'] = ParagraphStyle(
        'section', fontName='Helvetica-Bold', fontSize=7,
        textColor=P_MUTED, spaceBefore=14, spaceAfter=6,
        leading=9, letterSpacing=1.5,
    )
    styles['panel_title'] = ParagraphStyle(
        'panel_title', fontName='Helvetica', fontSize=6.5,
        textColor=P_CAPTION, spaceBefore=0, spaceAfter=5,
        leading=8, letterSpacing=1.2,
    )
    styles['body'] = ParagraphStyle(
        'body', fontName='Helvetica', fontSize=8.5,
        textColor=P_MUTED, spaceBefore=0, spaceAfter=0,
        leading=13,
    )
    styles['mono'] = ParagraphStyle(
        'mono', fontName='Courier', fontSize=7.5,
        textColor=P_MUTED, spaceBefore=0, spaceAfter=0, leading=11,
    )
    styles['caption'] = ParagraphStyle(
        'caption', fontName='Helvetica', fontSize=6.5,
        textColor=P_CAPTION, spaceBefore=0, spaceAfter=0, leading=9,
    )
    styles['kpi_label'] = ParagraphStyle(
        'kpi_label', fontName='Helvetica', fontSize=6,
        textColor=P_CAPTION, spaceBefore=0, spaceAfter=2,
        leading=7.5, letterSpacing=1.0,
    )
    styles['kpi_value'] = ParagraphStyle(
        'kpi_value', fontName='Times-Bold', fontSize=18,
        textColor=P_TEXT, spaceBefore=0, spaceAfter=2, leading=20,
    )
    styles['kpi_sub'] = ParagraphStyle(
        'kpi_sub', fontName='Helvetica', fontSize=6.5,
        textColor=P_MUTED, spaceBefore=0, spaceAfter=0, leading=8,
    )
    styles['badge_good'] = ParagraphStyle(
        'badge_good', fontName='Helvetica-Bold', fontSize=5.5,
        textColor=P_GOOD, spaceBefore=2, spaceAfter=0, leading=7,
    )
    styles['badge_warn'] = ParagraphStyle(
        'badge_warn', fontName='Helvetica-Bold', fontSize=5.5,
        textColor=P_WARN, spaceBefore=2, spaceAfter=0, leading=7,
    )
    styles['badge_bad'] = ParagraphStyle(
        'badge_bad', fontName='Helvetica-Bold', fontSize=5.5,
        textColor=P_BAD, spaceBefore=2, spaceAfter=0, leading=7,
    )
    styles['th'] = ParagraphStyle(
        'th', fontName='Helvetica-Bold', fontSize=6.5,
        textColor=P_CAPTION, spaceBefore=0, spaceAfter=0, leading=8,
    )
    styles['td'] = ParagraphStyle(
        'td', fontName='Helvetica', fontSize=7.5,
        textColor=P_TEXT, spaceBefore=0, spaceAfter=0, leading=9.5,
    )
    styles['td_mono'] = ParagraphStyle(
        'td_mono', fontName='Courier', fontSize=7,
        textColor=P_TEXT, spaceBefore=0, spaceAfter=0, leading=9,
    )
    styles['td_good'] = ParagraphStyle(
        'td_good', fontName='Courier-Bold', fontSize=7,
        textColor=P_GOOD, spaceBefore=0, spaceAfter=0, leading=9,
    )
    styles['td_warn'] = ParagraphStyle(
        'td_warn', fontName='Courier-Bold', fontSize=7,
        textColor=P_WARN, spaceBefore=0, spaceAfter=0, leading=9,
    )
    styles['td_bad'] = ParagraphStyle(
        'td_bad', fontName='Courier-Bold', fontSize=7,
        textColor=P_BAD, spaceBefore=0, spaceAfter=0, leading=9,
    )
    return styles


def _badge_style(score: float, styles):
    if score >= 0.60: return styles['badge_good']
    if score >= 0.40: return styles['badge_warn']
    return styles['badge_bad']


def _num_style(score: float, styles):
    if score >= 0.60: return styles['td_good']
    if score >= 0.40: return styles['td_warn']
    return styles['td_bad']


# ---------------------------------------------------------------------------
# Page header / footer callback
# ---------------------------------------------------------------------------

def _make_page_template(title: str, ts: str):
    """Returns an onPage callback drawing the header rule and footer on every page."""
    W, H = A4

    def on_page(canvas, doc):
        canvas.saveState()
        # Footer rule + text
        canvas.setStrokeColor(P_BORDER2)
        canvas.setLineWidth(0.5)
        canvas.line(15*mm, 12*mm, W - 15*mm, 12*mm)
        canvas.setFont('Helvetica', 6)
        canvas.setFillColor(P_CAPTION)
        canvas.drawString(15*mm, 9*mm, 'EvalSuite  ·  Legal Document Summarization')
        canvas.drawRightString(W - 15*mm, 9*mm,
                               f'{ts}  ·  Page {doc.page}')
        canvas.restoreState()

    return on_page


# ---------------------------------------------------------------------------
# Flowable: inline PNG chart
# ---------------------------------------------------------------------------

def _img(png_bytes: bytes, width_mm: float, height_mm: float = None) -> Image:
    buf = io.BytesIO(png_bytes)
    w   = width_mm * mm
    img = Image(buf, width=w, height=height_mm * mm if height_mm else None)
    if height_mm is None:
        # Let aspect ratio determine height naturally
        img._restrictSize(w, 500 * mm)
    return img


# ---------------------------------------------------------------------------
# KPI strip table
# ---------------------------------------------------------------------------

def _kpi_strip(d: dict, styles) -> Table:
    """Returns a wide Table acting as the KPI card strip."""
    specs = [
        ("ROUGE-1 F1",      d['rouge1_f1'],         "Unigram overlap"),
        ("ROUGE-2 F1",      d['rouge2_f1'],          "Bigram overlap"),
        ("ROUGE-L F1",      d['rougeL_f1'],          "Longest common subseq."),
        ("BERTScore F1",    d['bert_f1'],            "Semantic similarity"),
        ("Compression",     d['compression_ratio'],  "Ref / Gen ratio"),
        ("Gen Words",       float(d['summary_length']),   "words generated"),
        ("Ref Words",       float(d['reference_length']), "words in reference"),
    ]

    def card(label, value, sub, is_ratio=False):
        score = value if not is_ratio else -1  # skip badge for ratio/lengths
        ql    = quality_label(score) if score >= 0 and score <= 1 else ''
        bstyle = _badge_style(score, styles) if ql else None
        cell  = [
            Paragraph(label.upper(), styles['kpi_label']),
            Paragraph(
                f'{value:.2f}x' if is_ratio else
                (str(int(value)) if value == int(value) else f'{value:.3f}'),
                styles['kpi_value']
            ),
            Paragraph(sub, styles['kpi_sub']),
        ]
        if bstyle:
            cell.append(Paragraph(ql.upper(), bstyle))
        return cell

    cells = []
    for i, (lbl, val, sub) in enumerate(specs):
        is_ratio = (i == 4)
        is_int   = (i in (5, 6))
        cells.append(card(lbl, val, sub,
                          is_ratio=(is_ratio or is_int)))

    W = 180 * mm
    col_w = W / len(cells)
    tbl = Table([cells], colWidths=[col_w] * len(cells))
    tbl.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,-1), P_WHITE),
        ('BOX',         (0,0), (-1,-1), 0.5, P_BORDER2),
        ('INNERGRID',   (0,0), (-1,-1), 0.5, P_BORDER2),
        ('VALIGN',      (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING',  (0,0), (-1,-1), 7),
        ('BOTTOMPADDING',(0,0),(-1,-1), 7),
        ('LEFTPADDING', (0,0), (-1,-1), 7),
        ('RIGHTPADDING',(0,0), (-1,-1), 5),
    ]))
    return tbl


# ---------------------------------------------------------------------------
# ROUGE + BERTScore table
# ---------------------------------------------------------------------------

def _rouge_table(d: dict, styles) -> Table:
    header = [
        Paragraph('METRIC', styles['th']),
        Paragraph('PRECISION', styles['th']),
        Paragraph('RECALL', styles['th']),
        Paragraph('F1', styles['th']),
        Paragraph('QUALITY', styles['th']),
    ]
    rows = [header]
    specs = [
        ('ROUGE-1',    'rouge1'),
        ('ROUGE-2',    'rouge2'),
        ('ROUGE-L',    'rougeL'),
        ('ROUGE-Lsum', 'rougeLsum'),
    ]
    for name, key in specs:
        f1 = d[key + '_f1']
        nstyle = _num_style(f1, styles)
        rows.append([
            Paragraph(name, styles['td']),
            Paragraph(f"{d[key+'_precision']:.4f}", styles['td_mono']),
            Paragraph(f"{d[key+'_recall']:.4f}",    styles['td_mono']),
            Paragraph(f"{f1:.4f}", nstyle),
            Paragraph(quality_label(f1).upper(), _badge_style(f1, styles)),
        ])
    # BERTScore spacer row
    rows.append([
        Paragraph('', styles['td']),
        Paragraph('', styles['td']),
        Paragraph('', styles['td']),
        Paragraph('', styles['td']),
        Paragraph('', styles['td']),
    ])
    # BERTScore rows
    for lbl, key in [('BERT-P', 'bert_precision'),
                     ('BERT-R', 'bert_recall'),
                     ('BERT-F1','bert_f1')]:
        v = d[key]
        rows.append([
            Paragraph(lbl, styles['td']),
            Paragraph('—', styles['td_mono']),
            Paragraph('—', styles['td_mono']),
            Paragraph(f'{v:.4f}', _num_style(v, styles)),
            Paragraph(quality_label(v).upper(), _badge_style(v, styles)),
        ])

    col_w = [28*mm, 26*mm, 26*mm, 26*mm, 22*mm]
    tbl = Table(rows, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,0), P_BG),
        ('LINEBELOW',    (0,0), (-1,0), 0.8, P_BORDER2),
        ('LINEBELOW',    (0,4), (-1,4), 0.5, P_BORDER),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[P_WHITE, colors.HexColor('#fafaf9')]),
        ('GRID',         (0,0), (-1,-1), 0.4, P_BORDER),
        ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',   (0,0), (-1,-1), 5),
        ('BOTTOMPADDING',(0,0), (-1,-1), 5),
        ('LEFTPADDING',  (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    return tbl


# ---------------------------------------------------------------------------
# Confusion matrix table
# ---------------------------------------------------------------------------

def _cm_table(cm: dict, styles) -> Table:
    def cell(val, label, bg, br, tc):
        return [
            Paragraph(str(val),
                      ParagraphStyle('cv', fontName='Times-Bold', fontSize=16,
                                     textColor=tc, leading=18, alignment=TA_CENTER)),
            Paragraph(label.upper(),
                      ParagraphStyle('cl', fontName='Helvetica', fontSize=5.5,
                                     textColor=tc, leading=7, alignment=TA_CENTER)),
        ]

    hdr_sty = ParagraphStyle('ch', fontName='Helvetica', fontSize=6,
                              textColor=P_CAPTION, leading=8, alignment=TA_CENTER)

    data = [
        ['', Paragraph('PREDICTED +', hdr_sty), Paragraph('PREDICTED −', hdr_sty)],
        [Paragraph('ACTUAL +', hdr_sty),
         cell(cm['tp'], 'True Positive',  GOOD_BG, GOOD_BR, P_GOOD),
         cell(cm['fn'], 'False Negative', WARN_BG, WARN_BR, P_WARN)],
        [Paragraph('ACTUAL −', hdr_sty),
         cell(cm['fp'], 'False Positive', BAD_BG,  BAD_BR,  P_BAD),
         cell(cm['tn'], 'True Neg. (est)', colors.HexColor('#eff6ff'),
              colors.HexColor('#bfdbfe'), colors.HexColor('#1e40af'))],
    ]

    cw = [22*mm, 36*mm, 36*mm]
    tbl = Table(data, colWidths=cw, rowHeights=[8*mm, 20*mm, 20*mm])
    tbl.setStyle(TableStyle([
        ('GRID',          (0,0), (-1,-1), 0.5, P_BORDER2),
        ('BACKGROUND',    (1,1), (1,1),  GOOD_BG),
        ('BACKGROUND',    (2,1), (2,1),  WARN_BG),
        ('BACKGROUND',    (1,2), (1,2),  BAD_BG),
        ('BACKGROUND',    (2,2), (2,2),  colors.HexColor('#eff6ff')),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING',    (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    return tbl


def _cm_stats_table(cm: dict, styles) -> Table:
    items = [
        ('Precision', cm['precision'], P_GOOD),
        ('Recall',    cm['recall'],    P_WARN),
        ('F1 Score',  cm['f1'],        P_BLUE),
        ('Accuracy',  cm['accuracy'],  P_TEAL),
    ]
    cells = []
    for lbl, val, col in items:
        cells.append([
            Paragraph(f'{val:.3f}',
                      ParagraphStyle('sv', fontName='Times-Bold', fontSize=13,
                                     textColor=col, leading=15, alignment=TA_CENTER)),
            Paragraph(lbl.upper(),
                      ParagraphStyle('sl', fontName='Helvetica', fontSize=5.5,
                                     textColor=P_CAPTION, leading=7, alignment=TA_CENTER)),
        ])
    cw = [23.5*mm] * 4
    tbl = Table([cells], colWidths=cw)
    tbl.setStyle(TableStyle([
        ('BOX',          (0,0), (-1,-1), 0.5, P_BORDER2),
        ('INNERGRID',    (0,0), (-1,-1), 0.5, P_BORDER2),
        ('BACKGROUND',   (0,0), (-1,-1), P_WHITE),
        ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN',        (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING',   (0,0), (-1,-1), 6),
        ('BOTTOMPADDING',(0,0), (-1,-1), 6),
    ]))
    return tbl


# ---------------------------------------------------------------------------
# Section rule helper
# ---------------------------------------------------------------------------

def _section(title: str, styles) -> List:
    return [
        Spacer(1, 6*mm),
        Paragraph(title.upper(), styles['section']),
        HRFlowable(width='100%', thickness=0.8, color=P_BORDER2, spaceAfter=4),
    ]


# ---------------------------------------------------------------------------
# Main PDF generator — single evaluation
# ---------------------------------------------------------------------------

def generate_pdf_report(
        metrics: EvaluationMetrics,
        generated_summary: str,
        reference_summary: str,
        output_path: Path,
        title: str = "Summary Evaluation Report",
) -> None:
    logger.info("Building PDF report...")
    d   = metrics.to_dict()
    cm  = build_confusion_matrix(generated_summary, reference_summary)
    ts  = time.strftime("%d %B %Y, %H:%M")
    dev = "CUDA" if torch.cuda.is_available() else "CPU"
    styles = _make_styles()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=14*mm, bottomMargin=20*mm,
        title=title,
        author="EvalSuite",
    )

    story = []

    # ── Cover / masthead ──────────────────────────────────────────────────
    story.append(Paragraph("EVALSUITE  ·  LEGAL DOCUMENT SUMMARIZATION", styles['eyebrow']))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(title, styles['title']))
    story.append(HRFlowable(width='100%', thickness=2, color=P_TEXT, spaceAfter=3))
    meta_tbl = Table(
        [[Paragraph(f'Generated: {ts}  ·  Device: {dev}  ·  Eval time: {d["evaluation_time"]:.2f}s',
                    styles['caption'])]],
        colWidths=[180*mm],
    )
    meta_tbl.setStyle(TableStyle([
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING',(0,0), (-1,-1), 0),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 5*mm))

    # ── KPI strip ─────────────────────────────────────────────────────────
    story += _section("Key Performance Indicators", styles)
    story.append(_kpi_strip(d, styles))

    # ── Score tables + charts (side by side) ──────────────────────────────
    story += _section("Score Breakdown", styles)

    # Build chart PNGs
    png_pr    = chart_pr_grouped(d)
    png_f1    = chart_f1_horizontal(d)
    png_radar = chart_radar(d)
    png_donut = chart_length_donut(d['summary_length'], d['reference_length'])

    # Left: ROUGE+BERT table   Right: Radar
    left_cell  = [_rouge_table(d, styles)]
    right_cell = [
        Paragraph("RADAR — ALL DIMENSIONS", styles['panel_title']),
        _img(png_radar, 76),
    ]
    two_col = Table([[left_cell, right_cell]],
                    colWidths=[105*mm, 80*mm])
    two_col.setStyle(TableStyle([
        ('VALIGN',       (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING',  (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING',   (0,0), (-1,-1), 0),
        ('BOTTOMPADDING',(0,0), (-1,-1), 0),
        ('INNERGRID',    (0,0), (-1,-1), 0, colors.white),
    ]))
    story.append(two_col)
    story.append(Spacer(1, 4*mm))

    # Charts row: PR grouped + F1 horizontal + Donut
    story += _section("Score Distribution", styles)
    c1 = [Paragraph("PRECISION VS. RECALL", styles['panel_title']),
          _img(png_pr, 57, 33)]
    c2 = [Paragraph("F1 SCORE COMPARISON", styles['panel_title']),
          _img(png_f1, 57, 33)]
    c3 = [Paragraph("SUMMARY LENGTH", styles['panel_title']),
          _img(png_donut, 57, 33)]
    chart_row = Table([[c1, c2, c3]], colWidths=[60*mm, 60*mm, 60*mm])
    chart_row.setStyle(TableStyle([
        ('VALIGN',       (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING',  (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 3*mm),
        ('TOPPADDING',   (0,0), (-1,-1), 0),
        ('BOTTOMPADDING',(0,0), (-1,-1), 0),
    ]))
    story.append(chart_row)

    # ── Confusion matrix ──────────────────────────────────────────────────
    story.append(PageBreak())
    story += _section("Word-Level Confusion Matrix", styles)
    story.append(Paragraph(
        "Based on unique word-set overlap.  "
        "TP = in both  ·  FP = generated only  ·  FN = reference only  ·  TN = in neither (est.)",
        styles['caption']
    ))
    story.append(Spacer(1, 3*mm))

    png_bubble = chart_confusion_bubble(cm)
    cm_left  = [_cm_table(cm, styles),
                Spacer(1, 3*mm),
                _cm_stats_table(cm, styles)]
    cm_right = [Paragraph("BUBBLE CHART", styles['panel_title']),
                _img(png_bubble, 82, 62)]
    cm_row = Table([[cm_left, cm_right]], colWidths=[98*mm, 87*mm])
    cm_row.setStyle(TableStyle([
        ('VALIGN',       (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING',  (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING',   (0,0), (-1,-1), 0),
        ('BOTTOMPADDING',(0,0), (-1,-1), 0),
    ]))
    story.append(cm_row)

    # ── Summary previews ──────────────────────────────────────────────────
    story += _section("Summary Preview", styles)

    def text_panel(label, text, word_count, tag_color):
        wrapped = textwrap.fill(text[:800], width=90)
        return [
            Paragraph(label, ParagraphStyle(
                'tag', fontName='Helvetica-Bold', fontSize=6,
                textColor=tag_color, leading=8, letterSpacing=0.8)),
            Spacer(1, 2*mm),
            Paragraph(wrapped, styles['body']),
            Spacer(1, 2*mm),
            Paragraph(f'{word_count} words', styles['caption']),
        ]

    gen_panel = text_panel("GENERATED SUMMARY", generated_summary,
                           d['summary_length'], P_BLUE)
    ref_panel = text_panel("REFERENCE SUMMARY", reference_summary,
                           d['reference_length'], P_GOOD)

    preview_row = Table([[gen_panel, ref_panel]], colWidths=[88*mm, 88*mm])
    preview_row.setStyle(TableStyle([
        ('BOX',          (0,0), (0,0), 0.5, colors.HexColor('#bfdbfe')),
        ('BOX',          (1,0), (1,0), 0.5, GOOD_BR),
        ('BACKGROUND',   (0,0), (0,0), colors.HexColor('#eff6ff')),
        ('BACKGROUND',   (1,0), (1,0), GOOD_BG),
        ('VALIGN',       (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING',  (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING',   (0,0), (-1,-1), 8),
        ('BOTTOMPADDING',(0,0), (-1,-1), 8),
        ('INNERGRID',    (0,0), (-1,-1), 0, colors.white),
    ]))
    story.append(preview_row)

    # ── Build ─────────────────────────────────────────────────────────────
    cb = _make_page_template(title, ts)
    doc.build(story, onFirstPage=cb, onLaterPages=cb)
    logger.info(f"PDF saved to: {output_path}")


# ---------------------------------------------------------------------------
# Batch PDF report
# ---------------------------------------------------------------------------

def generate_batch_pdf_report(
        all_metrics: List[EvaluationMetrics],
        aggregate: Dict,
        output_path: Path,
        title: str = "Batch Evaluation Report",
) -> None:
    logger.info("Building batch PDF report...")
    ts = time.strftime("%d %B %Y, %H:%M")
    n  = len(all_metrics)
    styles = _make_styles()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=14*mm, bottomMargin=20*mm,
        title=title, author="EvalSuite",
    )

    story = []

    # Masthead
    story.append(Paragraph("EVALSUITE  ·  LEGAL DOCUMENT SUMMARIZATION", styles['eyebrow']))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(title, styles['title']))
    story.append(HRFlowable(width='100%', thickness=2, color=P_TEXT, spaceAfter=3))
    story.append(Paragraph(
        f'{n} summaries evaluated  ·  {ts}',
        styles['caption']))
    story.append(Spacer(1, 5*mm))

    # Aggregate KPI strip
    def ag(key): return aggregate.get(key, {})

    agg_specs = [
        ("ROUGE-1 F1",   ag('rouge1_f1'),   "mean"),
        ("ROUGE-2 F1",   ag('rouge2_f1'),   "mean"),
        ("ROUGE-L F1",   ag('rougeL_f1'),   "mean"),
        ("BERT F1",      ag('bert_f1'),      "mean"),
        ("Evaluated",    {'mean': float(n), 'std': 0}, "summaries"),
    ]
    kpi_cells = []
    for lbl, stats, sub in agg_specs:
        mean = stats.get('mean', 0)
        std  = stats.get('std',  0)
        sub_txt = f'± {std:.3f} std' if sub == 'mean' else sub
        kpi_cells.append([
            Paragraph(lbl.upper(), styles['kpi_label']),
            Paragraph(
                str(int(mean)) if lbl == 'Evaluated'
                else f'{mean:.3f}',
                styles['kpi_value']
            ),
            Paragraph(sub_txt, styles['kpi_sub']),
        ])

    story += _section("Aggregate Statistics", styles)
    cw = [180*mm / len(kpi_cells)] * len(kpi_cells)
    kpi_tbl = Table([kpi_cells], colWidths=cw)
    kpi_tbl.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,-1), P_WHITE),
        ('BOX',          (0,0), (-1,-1), 0.5, P_BORDER2),
        ('INNERGRID',    (0,0), (-1,-1), 0.5, P_BORDER2),
        ('VALIGN',       (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING',   (0,0), (-1,-1), 7),
        ('BOTTOMPADDING',(0,0), (-1,-1), 7),
        ('LEFTPADDING',  (0,0), (-1,-1), 7),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(kpi_tbl)

    # Per-summary table
    story += _section("Per-Summary Results", styles)

    def qs(score, val, styles=styles):
        return Paragraph(f'{val:.4f}', _num_style(score, styles))

    header = [
        Paragraph(h, styles['th']) for h in
        ['#', 'R1-F1', 'R2-F1', 'RL-F1', 'RLsum-F1', 'BERT-F1', 'Gen', 'Ref', 'Ratio']
    ]
    rows = [header]
    for i, m in enumerate(all_metrics):
        d = m.to_dict()
        rows.append([
            Paragraph(str(i+1), styles['td_mono']),
            qs(d['rouge1_f1'],   d['rouge1_f1']),
            qs(d['rouge2_f1'],   d['rouge2_f1']),
            qs(d['rougeL_f1'],   d['rougeL_f1']),
            Paragraph(f"{d['rougeLsum_f1']:.4f}", styles['td_mono']),
            qs(d['bert_f1'],     d['bert_f1']),
            Paragraph(str(d['summary_length']),   styles['td_mono']),
            Paragraph(str(d['reference_length']), styles['td_mono']),
            Paragraph(f"{d['compression_ratio']:.2f}x", styles['td_mono']),
        ])

    cw2 = [10*mm, 22*mm, 22*mm, 22*mm, 22*mm, 22*mm, 16*mm, 16*mm, 16*mm]
    tbl2 = Table(rows, colWidths=cw2, repeatRows=1)
    tbl2.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0),   P_BG),
        ('LINEBELOW',     (0,0), (-1,0),   0.8, P_BORDER2),
        ('ROWBACKGROUNDS',(0,1), (-1,-1),  [P_WHITE, colors.HexColor('#fafaf9')]),
        ('GRID',          (0,0), (-1,-1),  0.4, P_BORDER),
        ('VALIGN',        (0,0), (-1,-1),  'MIDDLE'),
        ('ALIGN',         (1,0), (-1,-1),  'RIGHT'),
        ('TOPPADDING',    (0,0), (-1,-1),  4),
        ('BOTTOMPADDING', (0,0), (-1,-1),  4),
        ('LEFTPADDING',   (0,0), (-1,-1),  5),
        ('RIGHTPADDING',  (0,0), (-1,-1),  5),
    ]))
    story.append(tbl2)

    # Trend charts
    story += _section("Trend Analysis", styles)
    png_trend = chart_trend(all_metrics)
    png_dist  = chart_mean_std(aggregate)

    c1 = [Paragraph("F1 SCORES PER SUMMARY", styles['panel_title']), _img(png_trend, 88, 40)]
    c2 = [Paragraph("MEAN F1 ± STD DEV",     styles['panel_title']), _img(png_dist,  88, 40)]
    trend_row = Table([[c1, c2]], colWidths=[91*mm, 91*mm])
    trend_row.setStyle(TableStyle([
        ('VALIGN',       (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING',  (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 3*mm),
        ('TOPPADDING',   (0,0), (-1,-1), 0),
        ('BOTTOMPADDING',(0,0), (-1,-1), 0),
    ]))
    story.append(trend_row)

    cb = _make_page_template(title, ts)
    doc.build(story, onFirstPage=cb, onLaterPages=cb)
    logger.info(f"Batch PDF saved to: {output_path}")


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_results(metrics: EvaluationMetrics, output_path: Path,
                 generated_summary: str, reference_summary: str) -> None:
    # JSON
    json_path = output_path.with_suffix('.json')
    json_path.write_text(json.dumps(metrics.to_dict(), indent=2), encoding='utf-8')
    logger.info(f"JSON saved to: {json_path}")
    # PDF
    pdf_path = output_path.with_suffix('.pdf')
    generate_pdf_report(metrics, generated_summary, reference_summary, pdf_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Legal Document Summaries — outputs a PDF report directly",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_summarizer.py generated.txt reference.txt
  python evaluate_summarizer.py generated.txt reference.txt --no-bertscore -o my_report
  python evaluate_summarizer.py --batch summaries_dir/ references_dir/
        """
    )
    parser.add_argument("generated", nargs='?', help="Generated summary file")
    parser.add_argument("reference", nargs='?', help="Reference summary file")
    parser.add_argument("--batch",        action="store_true",
                        help="Batch mode: pass two directories")
    parser.add_argument("--no-bertscore", action="store_true",
                        help="Skip BERTScore (faster)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path stem (no extension). Default: evaluation_results")
    parser.add_argument("--device", choices=['cuda', 'cpu'],
                        help="Device for BERTScore")
    args = parser.parse_args()

    if not args.generated or not args.reference:
        parser.print_help()
        sys.exit(1)

    try:
        evaluator = SummaryEvaluator(device=args.device)

        if args.batch:
            gen_dir = Path(args.generated)
            ref_dir = Path(args.reference)
            if not gen_dir.is_dir() or not ref_dir.is_dir():
                logger.error("Batch mode requires two directories"); sys.exit(1)
            gen_files = sorted(gen_dir.glob("*.txt"))
            ref_files = sorted(ref_dir.glob("*.txt"))
            generated_summaries = [f.read_text('utf-8').strip() for f in gen_files]
            reference_summaries  = [f.read_text('utf-8').strip() for f in ref_files]
            all_metrics, aggregate = evaluator.evaluate_batch(
                generated_summaries, reference_summaries,
                use_bertscore=not args.no_bertscore)
            out = Path(args.output) if args.output else Path("batch_evaluation_results")
            out.with_suffix('.json').write_text(
                json.dumps({'individual_metrics': [m.to_dict() for m in all_metrics],
                            'aggregate_statistics': aggregate}, indent=2), 'utf-8')
            generate_batch_pdf_report(all_metrics, aggregate, out.with_suffix('.pdf'))
            logger.info(f"\nDone. PDF report: {out.with_suffix('.pdf')}")

        else:
            gen_path = Path(args.generated)
            ref_path = Path(args.reference)
            if not gen_path.exists():
                logger.error(f"File not found: {gen_path}"); sys.exit(1)
            if not ref_path.exists():
                logger.error(f"File not found: {ref_path}"); sys.exit(1)

            generated = gen_path.read_text('utf-8').strip()
            reference  = ref_path.read_text('utf-8').strip()

            metrics = evaluator.evaluate(generated, reference,
                                         use_bertscore=not args.no_bertscore)
            metrics.print_summary()

            out = Path(args.output) if args.output else Path("evaluation_results")
            save_results(metrics, out, generated, reference)
            logger.info(f"\nDone. PDF report: {out.with_suffix('.pdf')}")

    except KeyboardInterrupt:
        logger.info("Interrupted"); sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback; traceback.print_exc(); sys.exit(1)


if __name__ == "__main__":
    main()
