"""
Flask Routes  (v3 — large-data safe)

Key improvements:
  - Batch DB commits in upload (1 commit per BATCH_COMMIT_SIZE rows, not per row)
  - Hard row cap on upload (MAX_UPLOAD_ROWS)
  - Text length guard on /analyze (MAX_MANUAL_CHARS)
  - Dashboard uses SQL aggregations — never loads input_text column
  - Source breakdown uses a single GROUP BY query
  - Word frequencies from stored top_keywords (no full reprocess)
  - Batch result uses paginated table (not all rows at once)
  - Export streams to file safely
"""

import io
import json
import logging
import os
import tempfile
from datetime import datetime
from collections import defaultdict

from flask import (
    Blueprint, render_template, request, redirect,
    url_for, flash, jsonify, send_file, current_app, session,
)
from werkzeug.utils import secure_filename
from sqlalchemy import func

from sentiment_app.extensions import db
from sentiment_app.models.database import AnalysisResult, BatchAnalysis, UploadedFile
from sentiment_app.nlp_engine import (
    analyze_text,
    get_word_frequencies_from_keywords,
    MAX_TEXT_CHARS,
)
from sentiment_app.document_parser import parse_document, allowed_file, MAX_ROWS
from sentiment_app.chart_utils import (
    generate_wordcloud_base64,
    polarity_histogram_base64,
)
from sentiment_app.groq_engine import analyze_with_groq
from sentiment_app.report_utils import build_batch_pdf_report, build_single_pdf_report

import pandas as pd

logger = logging.getLogger(__name__)
bp     = Blueprint('main', __name__)

# ── Tuneable limits ───────────────────────────────────────────────────────────
MAX_UPLOAD_ROWS    = MAX_ROWS       # from document_parser (5 000)
MAX_MANUAL_CHARS   = MAX_TEXT_CHARS # from nlp_engine     (50 000)
BATCH_COMMIT_SIZE  = 100            # commit every N rows during upload


# ── Helpers ───────────────────────────────────────────────────────────────────
def _save_result(result: dict, source_type='manual', source_filename=None, batch_id=None, groq_result=None):
    """Add a result to the session WITHOUT committing (caller batches commits)."""
    keywords = ', '.join(result.get('top_keywords', []))
    record = AnalysisResult(
        user_id               = session['user_id'],
        input_text            = result['input_text'],
        source_type           = source_type,
        source_filename       = source_filename,
        vader_sentiment       = result['vader']['sentiment'],
        textblob_sentiment    = result['textblob']['sentiment'],
        groq_sentiment        = (groq_result or {}).get('sentiment'),
        groq_explanation      = (groq_result or {}).get('explanation'),
        final_sentiment       = result['final_sentiment'],
        vader_compound        = result['vader']['compound'],
        vader_pos             = result['vader']['pos'],
        vader_neg             = result['vader']['neg'],
        vader_neu             = result['vader']['neu'],
        textblob_polarity     = result['textblob']['polarity'],
        textblob_subjectivity = result['textblob']['subjectivity'],
        word_count            = result.get('word_count', 0),
        sentence_count        = result.get('sentence_count', 0),
        top_keywords          = keywords,
        batch_id              = batch_id,
    )
    db.session.add(record)
    return record


def _save_and_commit(result: dict, source_type='manual', source_filename=None, batch_id=None, groq_result=None):
    """Convenience wrapper that also commits immediately (used for single analyses)."""
    record = _save_result(result, source_type, source_filename, batch_id, groq_result)
    db.session.commit()
    return record


def _owned_result(result_id):
    return AnalysisResult.query.filter_by(
        id=result_id,
        user_id=session['user_id'],
    ).first_or_404()


def _owned_batch(batch_id):
    return BatchAnalysis.query.filter_by(
        id=batch_id,
        user_id=session['user_id'],
    ).first_or_404()


# ── Home ──────────────────────────────────────────────────────────────────────
@bp.route('/')
def index():
    user_results = AnalysisResult.query.filter_by(user_id=session['user_id'])
    recent = user_results.order_by(AnalysisResult.created_at.desc()).limit(5).all()
    total  = user_results.count()
    # Single aggregation query instead of 3 separate COUNTs
    counts = dict(
        db.session.query(AnalysisResult.final_sentiment, func.count())
        .filter(AnalysisResult.user_id == session['user_id'])
        .group_by(AnalysisResult.final_sentiment)
        .all()
    )
    return render_template('index.html', recent=recent, stats={
        'total':    total,
        'positive': counts.get('positive', 0),
        'negative': counts.get('negative', 0),
        'neutral':  counts.get('neutral',  0),
        'mixed':    counts.get('mixed', 0),
    })


# ── Analyse (single text) ─────────────────────────────────────────────────────
@bp.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text', '').strip()
    if not text:
        flash('Please enter some text to analyse.', 'warning')
        return redirect(url_for('main.index'))

    # Server-side length guard
    if len(text) > MAX_MANUAL_CHARS:
        flash(
            f'Text is too long ({len(text):,} characters). '
            f'Maximum is {MAX_MANUAL_CHARS:,} characters. '
            f'The text has been trimmed to the limit.',
            'warning'
        )
        text = text[:MAX_MANUAL_CHARS]

    # In the textarea, each non-empty line is a separate manual input.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > MAX_UPLOAD_ROWS:
        flash(f'Enter no more than {MAX_UPLOAD_ROWS:,} non-empty lines at once.', 'warning')
        return redirect(url_for('main.index'))

    if len(lines) > 1:
        filename = f"manual_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        batch = BatchAnalysis(
            user_id=session['user_id'],
            filename=filename,
            file_type='manual',
            total_records=len(lines),
        )
        db.session.add(batch)
        db.session.flush()
        positive = negative = neutral = saved = 0

        for index, line in enumerate(lines, start=1):
            analysis = analyze_text(line)
            if 'error' not in analysis:
                record = _save_result(
                    analysis,
                    source_type='manual',
                    source_filename=filename,
                    batch_id=batch.id,
                )
                saved += 1
                if record.final_sentiment == 'positive':
                    positive += 1
                elif record.final_sentiment == 'negative':
                    negative += 1
                else:
                    neutral += 1

            if index % BATCH_COMMIT_SIZE == 0:
                db.session.commit()

        if not saved:
            db.session.rollback()
            flash('The entered lines could not be analysed.', 'danger')
            return redirect(url_for('main.index'))

        batch.positive_count = positive
        batch.negative_count = negative
        batch.neutral_count = neutral
        db.session.commit()
        flash(f'Analysed {saved} non-empty lines as separate inputs.', 'success')
        return redirect(url_for('main.batch_result', batch_id=batch.id))

    result = analyze_text(text)
    if 'error' in result:
        flash(result['error'], 'danger')
        return redirect(url_for('main.index'))

    groq_result = analyze_with_groq(text)
    record = _save_and_commit(result, source_type='manual', groq_result=groq_result)
    return redirect(url_for('main.result', result_id=record.id))


# ── Single Result ─────────────────────────────────────────────────────────────
@bp.route('/result/<int:result_id>')
def result(result_id):
    record        = _owned_result(result_id)
    # Re-run only the lightweight analyses needed for display
    # (pos_tags, sentences, emotions, readability are not stored in DB)
    nlp           = analyze_text(record.input_text)
    wordcloud_b64 = generate_wordcloud_base64(record.input_text)
    return render_template('result.html', record=record, nlp=nlp, wordcloud=wordcloud_b64)


@bp.route('/result/<int:result_id>/report.pdf')
def result_pdf_report(result_id):
    """Open or download a PDF report for one manually analyzed text."""
    record = _owned_result(result_id)
    report = build_single_pdf_report(record)
    return send_file(
        report,
        mimetype='application/pdf',
        as_attachment=request.args.get('download') == '1',
        download_name=f'sentiment_result_{record.id}_report.pdf',
    )


# ── Delete (POST only) ────────────────────────────────────────────────────────
@bp.route('/delete/<int:result_id>', methods=['POST'])
def delete_result(result_id):
    record = _owned_result(result_id)
    db.session.delete(record)
    db.session.commit()
    flash('Record deleted.', 'info')
    return redirect(url_for('main.history'))


# ── File Upload ───────────────────────────────────────────────────────────────
@bp.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('upload.html',
                               max_rows=MAX_UPLOAD_ROWS,
                               max_mb=current_app.config['MAX_CONTENT_LENGTH'] // (1024*1024))

    file = request.files.get('file')
    if not file or file.filename == '':
        flash('No file selected.', 'warning')
        return redirect(url_for('main.upload'))

    if not allowed_file(file.filename):
        flash('Unsupported file type. Allowed: TXT, PDF, DOCX, CSV, JSON, XLSX, XLS', 'danger')
        return redirect(url_for('main.upload'))

    filename = secure_filename(file.filename)
    if not filename:
        flash('The selected file name is invalid.', 'danger')
        return redirect(url_for('main.upload'))

    file_content = file.read()
    upload_dir = current_app.config['UPLOAD_FOLDER']
    os.makedirs(upload_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='wb',
        suffix=os.path.splitext(filename)[1],
        dir=upload_dir,
        delete=False,
    ) as temporary_file:
        filepath = temporary_file.name
        temporary_file.write(file_content)

    try:
        parse_result = parse_document(filepath, filename)
    finally:
        os.remove(filepath)

    texts = parse_result['texts']
    if not texts:
        flash(f'Could not extract text from "{filename}". Check the file.', 'danger')
        return redirect(url_for('main.upload'))

    if parse_result.get('capped'):
        flash(
            f'File has more than {MAX_UPLOAD_ROWS:,} records — '
            f'only the first {len(texts):,} were analysed.',
            'warning'
        )

    batch = BatchAnalysis(
        user_id       = session['user_id'],
        filename      = filename,
        file_type     = parse_result['file_type'],
        total_records = len(texts),
    )
    db.session.add(batch)
    db.session.flush()   # get batch.id before loop
    db.session.add(UploadedFile(
        user_id=session['user_id'],
        batch_id=batch.id,
        filename=filename,
        content_type=file.mimetype,
        size_bytes=len(file_content),
        content=file_content,
    ))

    pos_count = neg_count = neu_count = 0

    for i, text in enumerate(texts):
        analysis = analyze_text(text)
        if 'error' not in analysis:
            rec = _save_result(
                analysis,
                source_type     = 'file',
                source_filename = filename,
                batch_id        = batch.id,
            )
            if rec.final_sentiment == 'positive':   pos_count += 1
            elif rec.final_sentiment == 'negative': neg_count += 1
            else:                                    neu_count += 1

        # Commit in batches — avoids one fsync per row
        if (i + 1) % BATCH_COMMIT_SIZE == 0:
            db.session.commit()

    # Final commit for the tail + batch totals
    batch.positive_count = pos_count
    batch.negative_count = neg_count
    batch.neutral_count  = neu_count
    db.session.commit()

    flash(f'Successfully analysed {len(texts):,} records from "{filename}".', 'success')
    return redirect(url_for('main.batch_result', batch_id=batch.id))


@bp.route('/file/<int:file_id>')
def download_uploaded_file(file_id):
    uploaded_file = UploadedFile.query.filter_by(
        id=file_id,
        user_id=session['user_id'],
    ).first_or_404()
    return send_file(
        io.BytesIO(uploaded_file.content),
        mimetype=uploaded_file.content_type or 'application/octet-stream',
        as_attachment=True,
        download_name=uploaded_file.filename,
    )


# ── Batch Result (paginated) ──────────────────────────────────────────────────
@bp.route('/batch/<int:batch_id>')
def batch_result(batch_id):
    batch    = _owned_batch(batch_id)
    page     = request.args.get('page', 1, type=int)
    # Use dynamic relationship with pagination instead of loading all rows
    results_page = (
        AnalysisResult.query
        .filter_by(batch_id=batch_id, user_id=session['user_id'])
        .order_by(AnalysisResult.id)
        .paginate(page=page, per_page=50, error_out=False)
    )
    # Word cloud: use stored top_keywords (no full text join)
    kw_strings = [
        r.top_keywords for r in
        AnalysisResult.query.filter_by(batch_id=batch_id, user_id=session['user_id'])
        .with_entities(AnalysisResult.top_keywords)
        .limit(500)
        .all()
        if r.top_keywords
    ]
    freq_text = ' '.join(kw_strings)
    wc        = generate_wordcloud_base64(freq_text) if freq_text else ''

    return render_template('batch_result.html',
                           batch=batch,
                           results_page=results_page,
                           wordcloud=wc)


@bp.route('/batch/<int:batch_id>/report.pdf')
def batch_pdf_report(batch_id):
    """Download a summary and complete record appendix for one uploaded file."""
    batch = _owned_batch(batch_id)
    report = build_batch_pdf_report(batch, include_all_records=batch.file_type == 'manual')
    original_name = secure_filename(batch.filename) or f'batch_{batch_id}'
    report_name = f'{os.path.splitext(original_name)[0]}_sentiment_report.pdf'
    return send_file(
        report,
        mimetype='application/pdf',
        as_attachment=request.args.get('download') == '1',
        download_name=report_name,
    )


# ── Batch Export ──────────────────────────────────────────────────────────────
@bp.route('/batch/<int:batch_id>/export')
def batch_export(batch_id):
    batch = _owned_batch(batch_id)
    path  = os.path.join(current_app.config['UPLOAD_FOLDER'], f'batch_{batch_id}.csv')
    # Stream query to avoid loading all rows at once
    with open(path, 'w', encoding='utf-8', newline='') as f:
        first = True
        for chunk_start in range(0, batch.total_records + 1, 500):
            rows = (
                AnalysisResult.query
                .filter_by(batch_id=batch_id, user_id=session['user_id'])
                .offset(chunk_start)
                .limit(500)
                .all()
            )
            if not rows:
                break
            import csv
            for r in rows:
                d = r.to_dict()
                if first:
                    writer = csv.DictWriter(f, fieldnames=d.keys())
                    writer.writeheader()
                    first = False
                writer.writerow(d)
    return send_file(path, as_attachment=True, download_name=f'batch_{batch.filename}.csv')


# ── History ───────────────────────────────────────────────────────────────────
@bp.route('/history')
def history():
    page             = request.args.get('page', 1, type=int)
    sentiment_filter = request.args.get('sentiment', '')
    source_filter    = request.args.get('source', '')
    search_query     = request.args.get('q', '')

    query = AnalysisResult.query.filter_by(user_id=session['user_id'])
    if sentiment_filter: query = query.filter(AnalysisResult.final_sentiment == sentiment_filter)
    if source_filter:    query = query.filter(AnalysisResult.source_type == source_filter)
    if search_query:     query = query.filter(AnalysisResult.input_text.ilike(f'%{search_query}%'))

    pagination = query.order_by(AnalysisResult.created_at.desc()).paginate(
        page=page, per_page=15, error_out=False
    )
    batches = BatchAnalysis.query.filter_by(
        user_id=session['user_id']
    ).order_by(BatchAnalysis.created_at.desc()).limit(10).all()

    return render_template('history.html',
                           pagination=pagination,
                           batches=batches,
                           sentiment_filter=sentiment_filter,
                           source_filter=source_filter,
                           search_query=search_query)


# ── Dashboard ─────────────────────────────────────────────────────────────────
@bp.route('/dashboard')
def dashboard():
    user_results = AnalysisResult.query.filter_by(user_id=session['user_id'])
    total = user_results.count()
    mixed_count = AnalysisResult.query.filter(
        AnalysisResult.final_sentiment == 'mixed',
        AnalysisResult.user_id == session['user_id'],
    ).count()
    return render_template('dashboard.html', empty=(total == 0), total=total,
                           mixed_count=mixed_count)


# ── Dashboard JSON API ────────────────────────────────────────────────────────
@bp.route('/api/dashboard')
def api_dashboard():
    date_from = request.args.get('from')
    date_to   = request.args.get('to')

    # Build base query — select ONLY the scalar columns needed, NOT input_text
    base = db.session.query(
        AnalysisResult.id,
        AnalysisResult.final_sentiment,
        AnalysisResult.vader_sentiment,
        AnalysisResult.textblob_sentiment,
        AnalysisResult.vader_compound,
        AnalysisResult.textblob_subjectivity,
        AnalysisResult.created_at,
        AnalysisResult.source_type,
        AnalysisResult.top_keywords,
    ).filter(AnalysisResult.user_id == session['user_id'])

    try:
        if date_from:
            base = base.filter(AnalysisResult.created_at >= datetime.strptime(date_from, '%Y-%m-%d'))
        if date_to:
            dt_to = datetime.strptime(date_to, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
            base  = base.filter(AnalysisResult.created_at <= dt_to)
    except ValueError:
        pass

    rows = base.order_by(AnalysisResult.created_at).all()

    if not rows:
        return jsonify({'empty': True})

    total = len(rows)

    # ── Sentiment distribution (SQL-counted) ──────────────────────────────────
    sent_counts = {'positive': 0, 'negative': 0, 'neutral': 0, 'mixed': 0}
    for r in rows:
        if r.final_sentiment in sent_counts:
            sent_counts[r.final_sentiment] += 1

    # ── Trend per day ──────────────────────────────────────────────────────────
    daily = defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0, 'mixed': 0})
    for r in rows:
        if r.created_at and r.final_sentiment:
            day = r.created_at.strftime('%Y-%m-%d')
            if r.final_sentiment in daily[day]:
                daily[day][r.final_sentiment] += 1
    sorted_days = sorted(daily.keys())
    trend = {
        'dates':    sorted_days,
        'positive': [daily[d]['positive'] for d in sorted_days],
        'negative': [daily[d]['negative'] for d in sorted_days],
        'neutral':  [daily[d]['neutral']  for d in sorted_days],
        'mixed':    [daily[d]['mixed']    for d in sorted_days],
    }

    # ── Analyser comparison ────────────────────────────────────────────────────
    analyzers = {
        'VADER':       {'positive': 0, 'negative': 0, 'neutral': 0},
        'TextBlob':    {'positive': 0, 'negative': 0, 'neutral': 0},
    }
    for r in rows:
        for key, field in [('VADER', r.vader_sentiment),
                           ('TextBlob', r.textblob_sentiment)]:
            if field and field in analyzers[key]:
                analyzers[key][field] += 1
    analyzer_comparison = {
        'analyzers': list(analyzers.keys()),
        'positive':  [analyzers[a]['positive'] for a in analyzers],
        'negative':  [analyzers[a]['negative'] for a in analyzers],
        'neutral':   [analyzers[a]['neutral']  for a in analyzers],
    }

    # ── Word frequencies from stored top_keywords (no reprocessing!) ──────────
    kw_strings      = [r.top_keywords for r in rows if r.top_keywords]
    word_frequencies = get_word_frequencies_from_keywords(kw_strings)

    # ── Histogram (uses only vader_compound scalars) ───────────────────────────
    # Pass lightweight named-tuple rows (already fetched)
    histogram_b64 = polarity_histogram_base64(rows)

    # ── Averages ──────────────────────────────────────────────────────────────
    avg_compound    = round(sum(r.vader_compound or 0 for r in rows) / total, 3)
    avg_subjectivity= round(sum(r.textblob_subjectivity or 0 for r in rows) / total, 3)

    # ── Source breakdown — single GROUP BY ────────────────────────────────────
    src_q = (
        db.session.query(AnalysisResult.source_type, func.count())
        .filter(AnalysisResult.user_id == session['user_id'])
        .group_by(AnalysisResult.source_type)
        .all()
    )
    source_breakdown = {s: c for s, c in src_q}

    return jsonify({
        'empty':                  False,
        'total':                  total,
        'sentiment_distribution': {
            'labels': ['positive', 'negative', 'neutral', 'mixed'],
            'counts': [sent_counts['positive'], sent_counts['negative'], sent_counts['neutral'], sent_counts['mixed']],
            'colors': ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'],
        },
        'trend':                  trend,
        'analyzer_comparison':    analyzer_comparison,
        'word_frequencies':       word_frequencies,
        'histogram_b64':          histogram_b64,
        'avg_compound':           avg_compound,
        'avg_subjectivity':       avg_subjectivity,
        'source_breakdown': {
            'manual': source_breakdown.get('manual', 0),
            'file':   source_breakdown.get('file',   0),
            'api':    source_breakdown.get('api',    0),
        },
    })


# ── REST API ──────────────────────────────────────────────────────────────────
@bp.route('/api/analyze', methods=['POST'])
def api_analyze():
    payload = request.get_json(silent=True)
    if not payload or 'text' not in payload:
        return jsonify({'error': 'JSON body with "text" field required.'}), 400
    if not isinstance(payload['text'], str):
        return jsonify({'error': 'The "text" field must be a string.'}), 400
    text = payload['text'].strip()
    if not text:
        return jsonify({'error': 'Text is empty.'}), 400
    if len(text) > MAX_MANUAL_CHARS:
        text = text[:MAX_MANUAL_CHARS]

    result         = analyze_text(text)
    if 'error' in result:
        return jsonify({'error': result['error']}), 422
    groq_result    = analyze_with_groq(text)
    result['groq'] = groq_result
    record         = _save_and_commit(result, source_type='api', groq_result=groq_result)
    result['id']   = record.id
    result['pos_tags'] = [list(t) for t in result.get('pos_tags', [])]
    return jsonify(result)


# ── Export ────────────────────────────────────────────────────────────────────
@bp.route('/export/<fmt>')
def export_data(fmt):
    if fmt == 'csv':
        rows = AnalysisResult.query.filter_by(
            user_id=session['user_id']
        ).order_by(AnalysisResult.created_at.desc()).all()
        content = pd.DataFrame([r.to_dict() for r in rows]).to_csv(index=False).encode('utf-8')
        return send_file(
            io.BytesIO(content),
            mimetype='text/csv',
            as_attachment=True,
            download_name='sentiment_export.csv',
        )

    elif fmt == 'json':
        rows = AnalysisResult.query.filter_by(
            user_id=session['user_id']
        ).order_by(AnalysisResult.created_at.desc()).all()
        content = json.dumps(
            [r.to_dict() for r in rows],
            ensure_ascii=False,
            indent=2,
        ).encode('utf-8')
        return send_file(
            io.BytesIO(content),
            mimetype='application/json',
            as_attachment=True,
            download_name='sentiment_export.json',
        )

    flash('Unknown export format.', 'danger')
    return redirect(url_for('main.history'))


# ── About ─────────────────────────────────────────────────────────────────────
@bp.route('/about')
def about():
    total   = AnalysisResult.query.filter_by(user_id=session['user_id']).count()
    counts  = dict(
        db.session.query(AnalysisResult.final_sentiment, func.count())
        .filter(AnalysisResult.user_id == session['user_id'])
        .group_by(AnalysisResult.final_sentiment).all()
    )
    batches = BatchAnalysis.query.filter_by(user_id=session['user_id']).count()
    return render_template('about.html', stats={
        'total':   total,
        'pos':     counts.get('positive', 0),
        'neg':     counts.get('negative', 0),
        'batches': batches,
    })
