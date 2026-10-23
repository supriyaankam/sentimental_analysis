"""
NLP Core Engine  (v3 — large-data safe)
========================================
  Hard limits to prevent hangs on large inputs:
    MAX_TEXT_CHARS     = 50,000  chars  (text input guard)
    MAX_SENTENCES      = 200     (sentence-level VADER cap)
    MAX_PREPROCESS_TOK = 5,000   tokens (POS/lemma cap)
"""

import re
import string
import logging
from collections import Counter

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from textblob import TextBlob

# ── Optional packages ─────────────────────────────────────────────────────────
try:
    from nrclex import NRCLex
    _NRCLEX_AVAILABLE = True
except ImportError:
    _NRCLEX_AVAILABLE = False

try:
    import textstat as _textstat
    _TEXTSTAT_AVAILABLE = True
except ImportError:
    _TEXTSTAT_AVAILABLE = False

try:
    from langdetect import detect as _langdetect
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

# ── Hard limits ───────────────────────────────────────────────────────────────
MAX_TEXT_CHARS     = 50_000   # truncate input text beyond this
MAX_SENTENCES      = 200      # max sentences for per-sentence VADER
MAX_PREPROCESS_TOK = 5_000    # max tokens for POS/lemma (slow on huge texts)

# ── NLTK data ─────────────────────────────────────────────────────────────────
# NLTK data is provisioned once per environment. Downloading datasets at
# import time made app startup hang or emit network errors in offline setups.
logger = logging.getLogger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────
_vader      = SentimentIntensityAnalyzer()
_lemmatizer = WordNetLemmatizer()
_stemmer    = PorterStemmer()
_stop_words = set(stopwords.words('english'))


# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe_truncate(text: str, max_chars: int = MAX_TEXT_CHARS) -> tuple:
    """Return (truncated_text, was_truncated)."""
    if len(text) > max_chars:
        return text[:max_chars], True
    return text, False


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> dict:
    if not text or not text.strip():
        return {}

    # Work on a capped copy so large pastes don't stall tokenization
    work_text, truncated = _safe_truncate(text)

    sentences = sent_tokenize(work_text)

    cleaned = work_text.lower()
    cleaned = re.sub(r'http\S+|www\S+', '', cleaned)
    cleaned = re.sub(r'\S+@\S+', '', cleaned)
    cleaned = re.sub(r'[^a-zA-Z\s]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    tokens   = word_tokenize(cleaned)
    # Cap expensive per-token ops for huge documents
    filtered = [
        t for t in tokens[:MAX_PREPROCESS_TOK]
        if t not in _stop_words and t not in string.punctuation and len(t) > 1
    ]
    lemmas   = [_lemmatizer.lemmatize(t) for t in filtered]
    pos_tags = pos_tag(filtered)

    freq         = Counter(lemmas)
    top_keywords = [w for w, _ in freq.most_common(10)]

    return {
        'original_text':  text,
        'cleaned_text':   cleaned,
        'sentences':      sentences,
        'tokens':         tokens,
        'filtered_tokens':filtered,
        'lemmas':         lemmas,
        'pos_tags':       pos_tags,
        'top_keywords':   top_keywords,
        'word_count':     len(tokens),
        'sentence_count': len(sentences),
        'truncated':      truncated,
    }


# ── VADER ─────────────────────────────────────────────────────────────────────
def analyze_vader(text: str) -> dict:
    # VADER itself is fast on long text — no truncation needed here
    scores   = _vader.polarity_scores(text[:MAX_TEXT_CHARS])
    compound = scores['compound']
    label    = 'positive' if compound >= 0.05 else ('negative' if compound <= -0.05 else 'neutral')
    return {
        'sentiment': label,
        'compound':  round(compound, 4),
        'pos':       round(scores['pos'], 4),
        'neg':       round(scores['neg'], 4),
        'neu':       round(scores['neu'], 4),
    }


# ── TextBlob ──────────────────────────────────────────────────────────────────
def analyze_textblob(text: str) -> dict:
    blob         = TextBlob(text[:MAX_TEXT_CHARS])
    polarity     = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    label        = 'positive' if polarity > 0.05 else ('negative' if polarity < -0.05 else 'neutral')
    return {
        'sentiment':    label,
        'polarity':     round(polarity, 4),
        'subjectivity': round(subjectivity, 4),
    }


# ── Sentence-level VADER (capped) ─────────────────────────────────────────────
def analyze_sentences(sentences: list) -> list:
    """VADER per sentence — capped at MAX_SENTENCES to prevent timeout on books."""
    results = []
    for sent in sentences[:MAX_SENTENCES]:
        if not sent.strip():
            continue
        s = _vader.polarity_scores(sent)
        compound = s['compound']
        results.append({
            'text':      sent,
            'sentiment': 'positive' if compound >= 0.05 else ('negative' if compound <= -0.05 else 'neutral'),
            'compound':  round(compound, 4),
        })
    return results


# ── Emotion detection ─────────────────────────────────────────────────────────
def analyze_emotions(text: str) -> dict:
    if not _NRCLEX_AVAILABLE:
        return {}
    try:
        nrc = NRCLex()
        nrc.load_raw_text(text[:MAX_TEXT_CHARS])
        freqs    = nrc.affect_frequencies
        emotions = ['joy','sadness','anger','fear','surprise','disgust','trust','anticipation']
        result   = {e: round(freqs.get(e, 0.0), 4) for e in emotions}
        result['dominant'] = max(result, key=result.get) if any(v > 0 for v in result.values()) else 'none'
        return result
    except Exception as e:
        logger.warning(f"Emotion analysis failed: {e}")
        return {}


# ── Language detection ────────────────────────────────────────────────────────
def detect_language(text: str) -> dict:
    if not _LANGDETECT_AVAILABLE:
        return {'language': 'en', 'is_english': True, 'warning': None}
    try:
        lang    = _langdetect(text[:500])
        is_en   = (lang == 'en')
        warning = None if is_en else (
            f"Detected language: {lang.upper()}. Models are optimised for English — results may be less accurate."
        )
        return {'language': lang, 'is_english': is_en, 'warning': warning}
    except Exception:
        return {'language': 'unknown', 'is_english': True, 'warning': None}


# ── Readability ───────────────────────────────────────────────────────────────
def analyze_readability(text: str) -> dict:
    if not _TEXTSTAT_AVAILABLE:
        return {}
    try:
        fk_grade = _textstat.flesch_kincaid_grade(text[:MAX_TEXT_CHARS])
        fk_ease  = _textstat.flesch_reading_ease(text[:MAX_TEXT_CHARS])
        fog      = _textstat.gunning_fog(text[:MAX_TEXT_CHARS])
        if   fk_ease >= 80: level = 'Very Easy'
        elif fk_ease >= 60: level = 'Easy'
        elif fk_ease >= 50: level = 'Fairly Easy'
        elif fk_ease >= 30: level = 'Difficult'
        else:               level = 'Very Difficult'
        return {
            'flesch_kincaid_grade': round(fk_grade, 1),
            'flesch_reading_ease':  round(fk_ease, 1),
            'gunning_fog':          round(fog, 1),
            'reading_level':        level,
        }
    except Exception as e:
        logger.warning(f"Readability failed: {e}")
        return {}


# ── Combined sentiment ───────────────────────────────────────────────────────
def _combined_sentiment(vader: str, textblob: str) -> str:
    """Use both local analyzers; resolve disagreement in favor of VADER."""
    return Counter([vader, textblob]).most_common(1)[0][0]


def _sentiment_segments(text: str) -> list:
    """Split text into independently worded clauses for mixed sentiment checks."""
    # Cover common punctuation delimiters and contrast/conjunction words. Keeping
    # each clause intact lets VADER retain its negation and intensifier handling.
    parts = re.split(r"(?:[,;:|/\\]+|\b(?:but|however|although|though|yet|while|whereas)\b|[.!?]+)", text, flags=re.IGNORECASE)
    return [part.strip() for part in parts if part and part.strip()]


def _mixed_sentiment(text: str) -> bool:
    """True when distinct parts of the same input express both polarities."""
    segments = _sentiment_segments(text)
    # A single clause should continue to use the whole-text ensemble.
    if len(segments) < 2:
        return False
    labels = set()
    for segment in segments[:MAX_SENTENCES]:
        compound = _vader.polarity_scores(segment)['compound']
        # VADER misses some common food-quality words. Treat these as polarity
        # cues at clause level so a positive clause cannot hide a negative one.
        if re.search(
            r"\b(?:tasteless|flavou?rless|bland|unseasoned|undercooked|overcooked|"
            r"inedible|stale|spoiled|rotten)\b|"
            r"\b(?:has|have) no taste\b|\blacks? (?:any )?(?:taste|flavou?r)\b",
            segment,
            re.IGNORECASE,
        ):
            labels.add('negative')
        if compound >= 0.05:
            labels.add('positive')
        elif compound <= -0.05:
            labels.add('negative')
    return labels == {'positive', 'negative'}


# ── Master analysis ───────────────────────────────────────────────────────────
def analyze_text(text: str) -> dict:
    """Full pipeline — safe for texts up to MAX_TEXT_CHARS characters."""
    if not text or not text.strip():
        return {'error': 'Empty text provided'}

    # Hard limit — truncate silently; the caller can detect via 'truncated' flag
    text, was_truncated = _safe_truncate(text)

    nlp_data     = preprocess_text(text)
    vader_result = analyze_vader(text)
    tb_result    = analyze_textblob(text)
    final_sentiment = _combined_sentiment(
        vader_result['sentiment'],
        tb_result['sentiment'],
    )

    compound = vader_result['compound']
    if   compound >= 0.5:  intensity = 'Strongly Positive'
    elif compound >= 0.05: intensity = 'Mildly Positive'
    elif compound <= -0.5: intensity = 'Strongly Negative'
    elif compound <= -0.05:intensity = 'Mildly Negative'
    else:                  intensity = 'Neutral'

    sentence_results = analyze_sentences(nlp_data.get('sentences', []))
    if _mixed_sentiment(text):
        final_sentiment = 'mixed'
    emotion_result   = analyze_emotions(text)
    lang_result      = detect_language(text)
    readability      = analyze_readability(text)

    return {
        'input_text':      text,
        'word_count':      nlp_data.get('word_count', 0),
        'sentence_count':  nlp_data.get('sentence_count', 0),
        'top_keywords':    nlp_data.get('top_keywords', []),
        'pos_tags':        nlp_data.get('pos_tags', [])[:30],
        'sentences':       sentence_results,
        'vader':           vader_result,
        'textblob':        tb_result,
        'final_sentiment': final_sentiment,
        'intensity':       intensity,
        'emotions':        emotion_result,
        'language':        lang_result,
        'readability':     readability,
        'truncated':       was_truncated,
        'is_positive':     final_sentiment == 'positive',
        'is_negative':     final_sentiment == 'negative',
        'is_neutral':      final_sentiment == 'neutral',
        'is_mixed':        final_sentiment == 'mixed',
    }


# ── Batch ─────────────────────────────────────────────────────────────────────
def analyze_texts_batch(texts: list) -> list:
    return [analyze_text(t) for t in texts if isinstance(t, str) and t.strip()]


# ── Word frequencies (fast: uses stored top_keywords, no re-preprocessing) ────
def get_word_frequencies_from_keywords(keyword_strings: list) -> dict:
    """
    Build word frequency from already-stored top_keywords strings.
    Each string is comma-separated: 'love, product, amazing, quality'.
    Much faster than re-running preprocess_text() on all input texts.
    """
    counter = Counter()
    for kw_str in keyword_strings:
        if kw_str:
            for kw in kw_str.split(','):
                kw = kw.strip()
                if kw:
                    counter[kw] += 1
    return dict(counter.most_common(100))


# ── Legacy full-reprocess version (kept for single-text word cloud) ────────────
def get_word_frequencies(texts: list) -> dict:
    """Full reprocess — only use for small text lists (single result word clouds)."""
    all_lemmas = []
    for text in texts:
        if isinstance(text, str) and text.strip():
            d = preprocess_text(text)
            all_lemmas.extend(d.get('lemmas', []))
    return dict(Counter(all_lemmas).most_common(100))
