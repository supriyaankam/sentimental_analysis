"""Groq sentiment analysis for single-text requests."""

import json
import logging

from flask import current_app
from groq import Groq, GroqError

logger = logging.getLogger(__name__)


def analyze_with_groq(text: str):
    """Return a sentiment label and short explanation, or None if unavailable."""
    api_key = current_app.config.get('GROQ_API_KEY')
    if not api_key:
        return None

    try:
        response = Groq(api_key=api_key).chat.completions.create(
            model=current_app.config['GROQ_MODEL'],
            temperature=0,
            response_format={'type': 'json_object'},
            messages=[
                {
                    'role': 'system',
                    'content': (
                        'Classify the sentiment of the supplied text. Reply only with JSON '
                        'containing sentiment (positive, negative, or neutral) and a short explanation.'
                    ),
                },
                {'role': 'user', 'content': text[:5000]},
            ],
            timeout=20.0,
        )
        content = response.choices[0].message.content or '{}'
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError('Groq returned a non-object JSON response')
        sentiment = str(data.get('sentiment', '')).lower()
        if sentiment not in {'positive', 'negative', 'neutral'}:
            raise ValueError('Groq returned an invalid sentiment label')
        return {
            'sentiment': sentiment,
            'explanation': str(data.get('explanation', ''))[:1000],
        }
    except (GroqError, json.JSONDecodeError, ValueError, IndexError, AttributeError):
        logger.exception('Groq sentiment analysis failed')
        return {'sentiment': None, 'explanation': 'Groq analysis failed. Check the server log.'}
