"""
Chart Utilities  (v3 — large-data safe)
==========================================
  - Word cloud uses pre-tokenised keyword strings — no full reprocessing
  - polarity_histogram_base64 accepts both ORM objects and lightweight named tuples
  - Dark-themed word clouds
"""

import base64
import io
import logging
from collections import defaultdict, Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

logger = logging.getLogger(__name__)

DARK_BG    = '#0f0f1a'
CARD_BG    = '#1a1a2e'
TEXT_COLOR = '#94a3b8'


# ── Word Cloud ────────────────────────────────────────────────────────────────
def generate_wordcloud_base64(text: str, width: int = 900, height: int = 380) -> str:
    """Dark-background word cloud. `text` can be a pre-joined keyword string."""
    if not text or not text.strip():
        return ''
    try:
        wc = WordCloud(
            width=width,
            height=height,
            background_color=CARD_BG,
            stopwords=STOPWORDS,
            max_words=150,
            colormap='plasma',
            collocations=False,
            prefer_horizontal=0.85,
        ).generate(text)
        buf = io.BytesIO()
        wc.to_image().save(buf, format='PNG')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        logger.error(f'Wordcloud error: {e}')
        return ''


# ── Sentiment distribution ────────────────────────────────────────────────────
def sentiment_distribution_data(results) -> dict:
    counts = Counter(
        r.final_sentiment for r in results if r.final_sentiment
    )
    labels = ['positive', 'negative', 'neutral', 'mixed']
    return {
        'labels': labels,
        'counts': [counts.get(l, 0) for l in labels],
        'colors': ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'],
    }


# ── Trend ─────────────────────────────────────────────────────────────────────
def trend_data(results) -> dict:
    daily = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0, 'mixed': 0})
    for r in results:
        if r.created_at and r.final_sentiment:
            day = r.created_at.strftime('%Y-%m-%d')
            daily[day][r.final_sentiment] += 1
    sorted_days = sorted(daily.keys())
    return {
        'dates':    sorted_days,
        'positive': [daily[d]['positive'] for d in sorted_days],
        'negative': [daily[d]['negative'] for d in sorted_days],
        'neutral':  [daily[d]['neutral']  for d in sorted_days],
        'mixed':    [daily[d]['mixed']    for d in sorted_days],
    }


# ── Analyser comparison ───────────────────────────────────────────────────────
def analyzer_comparison_data(results) -> dict:
    analyzers = {
        'VADER':       {'positive': 0, 'negative': 0, 'neutral': 0},
        'TextBlob':    {'positive': 0, 'negative': 0, 'neutral': 0},
    }
    for r in results:
        for key, field in [('VADER', r.vader_sentiment),
                           ('TextBlob', r.textblob_sentiment)]:
            if field and field in analyzers[key]:
                analyzers[key][field] += 1
    return {
        'analyzers': list(analyzers.keys()),
        'positive':  [analyzers[a]['positive'] for a in analyzers],
        'negative':  [analyzers[a]['negative'] for a in analyzers],
        'neutral':   [analyzers[a]['neutral']  for a in analyzers],
    }


# ── Polarity Histogram ────────────────────────────────────────────────────────
def polarity_histogram_base64(results) -> str:
    """
    Accepts any iterable with a .vader_compound attribute.
    Works with both full ORM objects and lightweight query rows.
    """
    try:
        scores = [r.vader_compound for r in results if r.vader_compound is not None]
        if not scores:
            return ''

        fig, ax = plt.subplots(figsize=(8, 3), facecolor=CARD_BG)
        ax.set_facecolor(CARD_BG)

        n, bins, patches = ax.hist(scores, bins=25, edgecolor='none')
        for patch, left in zip(patches, bins[:-1]):
            if left >= 0.05:
                patch.set_facecolor('#2ecc71')
                patch.set_alpha(0.85)
            elif left <= -0.05:
                patch.set_facecolor('#e74c3c')
                patch.set_alpha(0.85)
            else:
                patch.set_facecolor('#f39c12')
                patch.set_alpha(0.85)

        ax.axvline(0, color=TEXT_COLOR, linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('VADER Compound Score', fontsize=10, color=TEXT_COLOR)
        ax.set_ylabel('Frequency', fontsize=10, color=TEXT_COLOR)
        ax.set_title('Sentiment Score Distribution', fontsize=12,
                     fontweight='bold', color='#f1f5f9', pad=10)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a2a3e')
            spine.set_alpha(0.6)

        plt.tight_layout(pad=1.2)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=CARD_BG)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        logger.error(f'Histogram error: {e}')
        return ''
