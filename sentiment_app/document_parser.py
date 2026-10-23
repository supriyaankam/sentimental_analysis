"""
Document Parser  (v3 — large-data safe)
=========================================
Row limits per format to prevent memory exhaustion:
  CSV / Excel  : MAX_ROWS = 5,000  rows
  TXT          : MAX_PARAGRAPHS = 2,000 blocks
  JSON         : MAX_STRINGS = 5,000 strings
  PDF          : all pages (PDF pages are already natural chunks)
  DOCX         : all paragraphs (DOCX paragraphs are small)
"""

import json
import logging
import os
import chardet
import pandas as pd
import PyPDF2
from docx import Document

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'csv', 'json', 'xlsx', 'xls'}

# Hard caps per format
MAX_ROWS       = 5_000    # CSV / Excel rows
MAX_PARAGRAPHS = 2_000    # TXT / DOCX paragraphs
MAX_STRINGS    = 5_000    # JSON extracted strings
MAX_TEXT_LEN   = 10_000   # chars per individual text block (truncate silently)


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_file_type(filename: str) -> str:
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else 'unknown'


def _cap_text(text: str) -> str:
    """Truncate a single text block to MAX_TEXT_LEN characters."""
    return text[:MAX_TEXT_LEN] if len(text) > MAX_TEXT_LEN else text


# ── TXT ───────────────────────────────────────────────────────────────────────
def parse_txt(filepath: str) -> list:
    try:
        with open(filepath, 'rb') as f:
            raw = f.read(20_000)   # sample for encoding detection only
        encoding = chardet.detect(raw).get('encoding', 'utf-8') or 'utf-8'

        texts = []
        with open(filepath, 'r', encoding=encoding, errors='replace') as f:
            current_para = []
            for line in f:
                if line.strip() == '':
                    if current_para:
                        para = ' '.join(current_para).strip()
                        if para:
                            texts.append(_cap_text(para))
                            if len(texts) >= MAX_PARAGRAPHS:
                                break
                        current_para = []
                else:
                    current_para.append(line.strip())

            # flush last paragraph
            if current_para and len(texts) < MAX_PARAGRAPHS:
                para = ' '.join(current_para).strip()
                if para:
                    texts.append(_cap_text(para))

        # fallback: if no blank-line paragraphs found, split by lines
        if not texts:
            with open(filepath, 'r', encoding=encoding, errors='replace') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(_cap_text(line))
                        if len(texts) >= MAX_PARAGRAPHS:
                            break

        return texts
    except Exception as e:
        logger.error(f"TXT parse error: {e}")
        return []


# ── PDF ───────────────────────────────────────────────────────────────────────
def parse_pdf(filepath: str) -> list:
    texts = []
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text and text.strip():
                    texts.append(_cap_text(text.strip()))
    except Exception as e:
        logger.error(f"PDF parse error: {e}")
    return texts


# ── DOCX ──────────────────────────────────────────────────────────────────────
def parse_docx(filepath: str) -> list:
    texts = []
    try:
        doc = Document(filepath)
        for para in doc.paragraphs:
            t = para.text.strip()
            if t:
                texts.append(_cap_text(t))
                if len(texts) >= MAX_PARAGRAPHS:
                    break
        if len(texts) < MAX_PARAGRAPHS:
            for table in doc.tables:
                for row in table.rows:
                    row_text = ' | '.join(c.text.strip() for c in row.cells if c.text.strip())
                    if row_text:
                        texts.append(_cap_text(row_text))
                        if len(texts) >= MAX_PARAGRAPHS:
                            break
    except Exception as e:
        logger.error(f"DOCX parse error: {e}")
    return texts


# ── CSV ───────────────────────────────────────────────────────────────────────
def parse_csv(filepath: str) -> list:
    texts = []
    try:
        with open(filepath, 'rb') as f:
            raw = f.read(20_000)
        encoding = chardet.detect(raw).get('encoding', 'utf-8') or 'utf-8'

        # nrows cap + chunksize for memory safety
        df = pd.read_csv(
            filepath,
            encoding=encoding,
            encoding_errors='replace',
            nrows=MAX_ROWS,
        )
        df.dropna(how='all', inplace=True)

        preferred = ['text','review','comment','content','feedback',
                     'description','message','body','sentence']
        text_col = next(
            (col for col in df.columns if col.lower().strip() in preferred),
            None
        )

        if text_col:
            texts = df[text_col].dropna().astype(str).tolist()
        else:
            str_cols = df.select_dtypes(include='object').columns.tolist()
            if str_cols:
                texts = df[str_cols].fillna('').apply(
                    lambda row: ' '.join(str(v) for v in row if str(v).strip()), axis=1
                ).tolist()
            else:
                texts = df.astype(str).apply(lambda r: ' '.join(r), axis=1).tolist()

        texts = [_cap_text(t.strip()) for t in texts if t.strip() and t.strip() != 'nan']
    except Exception as e:
        logger.error(f"CSV parse error: {e}")
    return texts


# ── JSON ──────────────────────────────────────────────────────────────────────
def parse_json(filepath: str) -> list:
    texts = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            data = json.load(f)

        preferred_keys = {'text','review','comment','content','feedback',
                          'message','body','description'}

        def extract_strings(obj):
            if len(texts) >= MAX_STRINGS:
                return
            if isinstance(obj, str):
                if obj.strip():
                    texts.append(_cap_text(obj.strip()))
            elif isinstance(obj, list):
                for item in obj:
                    if len(texts) >= MAX_STRINGS:
                        break
                    extract_strings(item)
            elif isinstance(obj, dict):
                found = False
                for key in preferred_keys:
                    if key in obj and isinstance(obj[key], str) and obj[key].strip():
                        texts.append(_cap_text(obj[key].strip()))
                        found = True
                        break
                if not found:
                    for v in obj.values():
                        if len(texts) >= MAX_STRINGS:
                            break
                        extract_strings(v)

        extract_strings(data)
    except Exception as e:
        logger.error(f"JSON parse error: {e}")
    return texts


# ── XLSX / XLS ────────────────────────────────────────────────────────────────
def parse_excel(filepath: str) -> list:
    texts = []
    try:
        xls = pd.ExcelFile(filepath)
        # Only process the first 3 sheets to avoid very large multi-sheet files
        for sheet in xls.sheet_names[:3]:
            if len(texts) >= MAX_ROWS:
                break
            remaining = MAX_ROWS - len(texts)
            df = pd.read_excel(xls, sheet_name=sheet, nrows=remaining)
            df.dropna(how='all', inplace=True)

            preferred = ['text','review','comment','content','feedback',
                         'description','message','body','sentence']
            text_col = next(
                (col for col in df.columns if str(col).lower().strip() in preferred),
                None
            )

            if text_col:
                sheet_texts = df[text_col].dropna().astype(str).tolist()
            else:
                str_cols = df.select_dtypes(include='object').columns.tolist()
                if str_cols:
                    sheet_texts = df[str_cols].fillna('').apply(
                        lambda row: ' '.join(str(v) for v in row if str(v).strip()), axis=1
                    ).tolist()
                else:
                    sheet_texts = df.astype(str).apply(lambda r: ' '.join(r), axis=1).tolist()

            texts.extend(
                _cap_text(t.strip()) for t in sheet_texts
                if t.strip() and t.strip() != 'nan'
            )
    except Exception as e:
        logger.error(f"Excel parse error: {e}")
    return texts


# ── Master dispatcher ─────────────────────────────────────────────────────────
def parse_document(filepath: str, filename: str) -> dict:
    file_type = get_file_type(filename)
    texts     = []
    error     = None

    try:
        if   file_type == 'txt':              texts = parse_txt(filepath)
        elif file_type == 'pdf':              texts = parse_pdf(filepath)
        elif file_type == 'docx':             texts = parse_docx(filepath)
        elif file_type == 'csv':              texts = parse_csv(filepath)
        elif file_type == 'json':             texts = parse_json(filepath)
        elif file_type in ('xlsx', 'xls'):   texts = parse_excel(filepath)
        else:                                  error = f"Unsupported file type: {file_type}"
    except Exception as e:
        error = str(e)
        logger.error(f"parse_document error ({filename}): {e}")

    capped = len(texts) >= MAX_ROWS
    return {
        'texts':        texts,
        'file_type':    file_type,
        'record_count': len(texts),
        'error':        error,
        'capped':       capped,   # True if row limit was hit
    }
