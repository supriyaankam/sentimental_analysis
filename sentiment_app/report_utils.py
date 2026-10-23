"""Create downloadable PDF reports for a single uploaded analysis batch."""

from collections import Counter
from datetime import datetime
from io import BytesIO
from itertools import islice
import re
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sentiment_app.models.database import AnalysisResult


SENTIMENT_COLORS = {
    'positive': '#2e9d63',
    'negative': '#d9534f',
    'neutral': '#e3a82b',
}


def _clean_text(value, limit=100):
    """Make arbitrary uploaded text safe and compact for a PDF table cell."""
    value = re.sub(r'\s+', ' ', str(value or '')).replace('$', '')
    if len(value) > limit:
        return value[:limit - 1].rstrip() + '…'
    return value


def _add_manual_line_appendix(pdf, results_query, total):
    """Add every manual line with the result from its independent analysis."""
    rows = results_query.with_entities(
        AnalysisResult.input_text,
        AnalysisResult.final_sentiment,
        AnalysisResult.vader_sentiment,
        AnalysisResult.textblob_sentiment,
        AnalysisResult.vader_compound,
        AnalysisResult.textblob_polarity,
        AnalysisResult.word_count,
    ).yield_per(100)
    report_lines = []
    for index, row in enumerate(rows, start=1):
        vader_score = f'{row.vader_compound:.3f}' if row.vader_compound is not None else 'n/a'
        blob_score = f'{row.textblob_polarity:.3f}' if row.textblob_polarity is not None else 'n/a'
        summary = (
            f"{index}. {str(row.final_sentiment or 'unavailable').upper()} | "
            f"VADER {row.vader_sentiment or 'n/a'} ({vader_score}) | "
            f"TextBlob {row.textblob_sentiment or 'n/a'} ({blob_score}) | "
            f"{row.word_count or 0} words"
        )
        report_lines.append((_clean_text(summary, 135), True))
        full_text = re.sub(r'\s+', ' ', str(row.input_text or '')).replace('$', '')
        wrapped_text = textwrap.wrap(
            full_text, width=100, break_long_words=True, break_on_hyphens=False
        ) or ['(empty)']
        report_lines.extend((f'    {line}', False) for line in wrapped_text)

    page_size = 52
    total_pages = (len(report_lines) + page_size - 1) // page_size
    rows = iter(report_lines)
    page = 1
    while chunk := list(islice(rows, page_size)):
        fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        fig.text(0.07, 0.95,
                 f'Manual line-by-line results - page {page} of {total_pages} ({total:,} inputs)',
                 fontsize=13, fontweight='bold', color='#172033')
        y = 0.90
        for line, is_heading in chunk:
            fig.text(0.07, y, line, fontsize=7.5 if is_heading else 7,
                     fontweight='bold' if is_heading else 'normal',
                     color='#334155' if is_heading else '#64748b')
            y -= 0.015
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        page += 1


def build_batch_pdf_report(batch, include_all_records=False):
    """Return a concise PDF summary and representative record examples."""
    results_query = (
        AnalysisResult.query
        .filter_by(batch_id=batch.id)
        .order_by(AnalysisResult.id)
    )
    aggregate = results_query.with_entities(
        AnalysisResult.id,
        AnalysisResult.final_sentiment,
        AnalysisResult.vader_sentiment,
        AnalysisResult.textblob_sentiment,
        AnalysisResult.vader_compound,
        AnalysisResult.textblob_polarity,
        AnalysisResult.textblob_subjectivity,
        AnalysisResult.word_count,
    ).all()

    total = len(aggregate)
    counts = Counter(row.final_sentiment for row in aggregate if row.final_sentiment)
    average = lambda field: (
        sum(getattr(row, field) for row in aggregate if getattr(row, field) is not None)
        / sum(getattr(row, field) is not None for row in aggregate)
        if any(getattr(row, field) is not None for row in aggregate) else None
    )

    keyword_counts = Counter()
    for (keywords,) in results_query.with_entities(AnalysisResult.top_keywords).yield_per(500):
        if keywords:
            keyword_counts.update(
                word.strip().lower()
                for word in keywords.split(',')
                if word.strip()
            )

    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        pdf.infodict().update({
            'Title': f'Sentiment Analysis Report - {batch.filename}',
            'Subject': 'Sentiment analysis report for one uploaded dataset',
            'Creator': 'SentimentIQ',
        })

        fig = plt.figure(figsize=(11.69, 8.27), facecolor='white')
        fig.suptitle('Sentiment Analysis Report', x=0.06, y=0.94,
                     ha='left', fontsize=21, fontweight='bold', color='#172033')
        fig.text(0.06, 0.885, _clean_text(batch.filename, 120),
                 fontsize=13, color='#475569')
        fig.text(
            0.06, 0.845,
            f"Uploaded format: {str(batch.file_type).upper()}    "
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            fontsize=9, color='#64748b',
        )

        # Summary and interpretation
        fig.text(0.06, 0.77, 'Dataset overview', fontsize=13,
                 fontweight='bold', color='#172033')
        overview = [
            f'Analyzed records: {total:,} of {int(batch.total_records or 0):,} uploaded',
            f"Positive: {counts.get('positive', 0):,}  "
            f"({(counts.get('positive', 0) / total * 100) if total else 0:.1f}%)",
            f"Negative: {counts.get('negative', 0):,}  "
            f"({(counts.get('negative', 0) / total * 100) if total else 0:.1f}%)",
            f"Neutral: {counts.get('neutral', 0):,}  "
            f"({(counts.get('neutral', 0) / total * 100) if total else 0:.1f}%)",
            f"Average VADER compound: {average('vader_compound'):.3f}"
            if average('vader_compound') is not None else 'Average VADER compound: unavailable',
            f"Average TextBlob polarity: {average('textblob_polarity'):.3f}"
            if average('textblob_polarity') is not None else 'Average TextBlob polarity: unavailable',
            f"Average TextBlob subjectivity: {average('textblob_subjectivity'):.3f}"
            if average('textblob_subjectivity') is not None else 'Average TextBlob subjectivity: unavailable',
            f"Average words per record: {average('word_count'):.1f}"
            if average('word_count') is not None else 'Average words per record: unavailable',
        ]
        fig.text(0.06, 0.72, '\n'.join(overview), va='top', fontsize=11,
                 linespacing=1.65, color='#334155')

        if total:
            pie_ax = fig.add_axes([0.63, 0.40, 0.31, 0.39])
            labels = ['Positive', 'Negative', 'Neutral']
            values = [counts.get(label.lower(), 0) for label in labels]
            nonzero = [(label, value) for label, value in zip(labels, values) if value]
            if nonzero:
                pie_ax.pie(
                    [value for _, value in nonzero],
                    labels=[label for label, _ in nonzero],
                    colors=[SENTIMENT_COLORS[label.lower()] for label, _ in nonzero],
                    autopct='%1.1f%%', startangle=90,
                    textprops={'fontsize': 9, 'color': '#172033'},
                )
                pie_ax.set_title('Final sentiment distribution', fontsize=12,
                                 fontweight='bold', color='#172033', pad=14)
                pie_ax.axis('equal')

        top_keywords = keyword_counts.most_common(12)
        fig.text(0.06, 0.33, 'Most frequent keywords', fontsize=12,
                 fontweight='bold', color='#172033')
        keyword_line = ', '.join(f'{word} ({count})' for word, count in top_keywords)
        fig.text(0.06, 0.295, _clean_text(keyword_line, 190) if keyword_line else 'No keywords available.',
                 fontsize=9.5, color='#334155', wrap=True)

        dominant = max(('positive', 'negative', 'neutral'), key=lambda name: counts.get(name, 0)) if total else None
        if dominant:
            share = counts.get(dominant, 0) / total * 100
            insight = f'{dominant.capitalize()} is the most common label ({share:.1f}% of analyzed records).'
        else:
            insight = 'No records were available for sentiment analysis.'
        fig.text(0.06, 0.22, 'Key finding', fontsize=12,
                 fontweight='bold', color='#172033')
        fig.text(0.06, 0.185, insight, fontsize=10, color='#334155')
        appendix_note = (
            'The appendix lists every line and its separate analysis.'
            if include_all_records else
            'The appendix shows examples; use the CSV export for every analyzed record.'
        )
        fig.text(
            0.06, 0.09,
            'Method: final labels are calculated by SentimentIQ from VADER and TextBlob. '
            + appendix_note,
            fontsize=8, color='#64748b', wrap=True,
        )
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        if include_all_records:
            _add_manual_line_appendix(pdf, results_query, total)

        # A few representative examples keep report generation fast for large files.
        examples = []
        sentiments = () if include_all_records else ('positive', 'negative', 'neutral')
        for sentiment in sentiments:
            ordering = AnalysisResult.vader_compound.asc()
            if sentiment == 'positive':
                ordering = AnalysisResult.vader_compound.desc()
            elif sentiment == 'neutral':
                ordering = AnalysisResult.id.asc()
            examples.extend(
                results_query.filter(AnalysisResult.final_sentiment == sentiment)
                .order_by(None).order_by(ordering).limit(3).all()
            )

        if examples:
            fig, ax = plt.subplots(figsize=(11.69, 8.27), facecolor='white')
            ax.axis('off')
            ax.set_title('Representative analyzed records (up to three per sentiment)',
                         loc='left', fontsize=14, fontweight='bold', color='#172033', pad=16)
            sample_rows = [[
                _clean_text(row.final_sentiment, 12).capitalize(),
                _clean_text(row.vader_sentiment, 12).capitalize(),
                _clean_text(row.textblob_sentiment, 12).capitalize(),
                f'{row.vader_compound:.3f}' if row.vader_compound is not None else '',
                f'{row.textblob_polarity:.3f}' if row.textblob_polarity is not None else '',
                str(row.word_count or 0),
                _clean_text(row.input_text, 100),
            ] for row in examples]
            table = ax.table(
                cellText=sample_rows,
                colLabels=['Final', 'VADER', 'TextBlob', 'VADER score',
                           'TB polarity', 'Words', 'Text preview'],
                colWidths=[0.10, 0.10, 0.10, 0.12, 0.12, 0.07, 0.39],
                cellLoc='left', colLoc='left', loc='upper center',
                bbox=[0, 0.42, 1, 0.45],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            for (row_index, _), cell in table.get_celld().items():
                cell.set_edgecolor('#dbe2ea')
                cell.set_linewidth(0.4)
                if row_index == 0:
                    cell.set_facecolor('#e9eef5')
                    cell.set_text_props(weight='bold', color='#172033')
                else:
                    cell.set_facecolor('#ffffff' if row_index % 2 else '#f7f9fc')
                    cell.set_text_props(color='#334155')
            fig.text(0.06, 0.30,
                     'These examples illustrate analyzed records. The accompanying CSV export '
                     'contains the complete record-level results.',
                     fontsize=9, color='#64748b', wrap=True)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    buffer.seek(0)
    return buffer


def build_single_pdf_report(record):
    """Return a PDF with one manual input's scores and complete entered text."""
    buffer = BytesIO()
    text = str(record.input_text or '').replace('$', '')
    wrapped_lines = []
    for paragraph in text.splitlines() or ['']:
        wrapped_lines.extend(
            textwrap.wrap(paragraph, width=105, break_long_words=True,
                          break_on_hyphens=False) or ['']
        )
    text_page_size = 44
    extra_pages = max(0, (max(0, len(wrapped_lines) - 25) + text_page_size - 1) // text_page_size)

    with PdfPages(buffer) as pdf:
        pdf.infodict().update({
            'Title': f'Sentiment Analysis Report - Result {record.id}',
            'Subject': 'Sentiment analysis for one manually entered text',
            'Creator': 'SentimentIQ',
        })
        fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
        fig.text(0.08, 0.94, 'Sentiment Analysis Report', fontsize=20,
                 fontweight='bold', color='#172033')
        fig.text(0.08, 0.89, f'Manual input result #{record.id}',
                 fontsize=12, color='#475569')
        fig.text(0.08, 0.83,
                 f'Final sentiment: {str(record.final_sentiment or "Unavailable").upper()}',
                 fontsize=15, fontweight='bold',
                 color=SENTIMENT_COLORS.get(record.final_sentiment, '#475569'))
        metrics = [
            f"VADER sentiment: {record.vader_sentiment or 'Unavailable'}",
            f"VADER compound: {record.vader_compound:.3f}" if record.vader_compound is not None else 'VADER compound: unavailable',
            f"VADER positive / negative / neutral: {record.vader_pos or 0:.3f} / {record.vader_neg or 0:.3f} / {record.vader_neu or 0:.3f}",
            f"TextBlob sentiment: {record.textblob_sentiment or 'Unavailable'}",
            f"TextBlob polarity / subjectivity: {record.textblob_polarity or 0:.3f} / {record.textblob_subjectivity or 0:.3f}",
            f"Words: {record.word_count or 0}    Sentences: {record.sentence_count or 0}",
            f"Keywords: {_clean_text(record.top_keywords, 180) or 'None'}",
            f'Entered text pages: {extra_pages + 1}',
        ]
        fig.text(0.08, 0.77, '\n'.join(metrics), va='top', fontsize=10,
                 linespacing=1.8, color='#334155')
        fig.text(0.08, 0.48, 'Entered text', fontsize=12,
                 fontweight='bold', color='#172033')
        first_chunk = wrapped_lines[:25]
        fig.text(0.08, 0.44, '\n'.join(first_chunk) or '(empty)',
                 va='top', fontsize=8.5, linespacing=1.4,
                 color='#334155', wrap=True)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        offset = len(first_chunk)
        page_number = 2
        while offset < len(wrapped_lines):
            chunk = wrapped_lines[offset:offset + text_page_size]
            fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
            fig.text(0.08, 0.95,
                     f'Entered text (page {page_number} of {extra_pages + 1})',
                     fontsize=13, fontweight='bold', color='#172033')
            fig.text(0.08, 0.90, '\n'.join(chunk), va='top', fontsize=8.5,
                     linespacing=1.45, color='#334155', wrap=True)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            offset += len(chunk)
            page_number += 1

    buffer.seek(0)
    return buffer
