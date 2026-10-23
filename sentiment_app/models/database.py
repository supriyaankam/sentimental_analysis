"""
Database models — v2 (with indexes for large-data performance)
"""

from datetime import datetime
from sentiment_app.extensions import db


class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, server_default='')
    email = db.Column(db.String(254), nullable=False, unique=True, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    auth_version = db.Column(db.Integer, nullable=False, default=0, server_default='0')
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f'<User id={self.id} email={self.email}>'


class PasswordResetToken(db.Model):
    __tablename__ = 'password_reset_tokens'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    token_hash = db.Column(db.String(64), nullable=False, unique=True, index=True)
    token_type = db.Column(db.String(10), nullable=False, default='email', server_default='email')
    expires_at = db.Column(db.DateTime, nullable=False, index=True)
    failed_attempts = db.Column(db.Integer, nullable=False, default=0, server_default='0')
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    user = db.relationship('User', backref=db.backref('password_reset_tokens', lazy='dynamic'))

    def __repr__(self):
        return f'<PasswordResetToken id={self.id} user_id={self.user_id}>'


class AnalysisResult(db.Model):
    __tablename__ = 'analysis_results'

    id                  = db.Column(db.Integer, primary_key=True)
    user_id             = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    input_text          = db.Column(db.Text, nullable=False)
    source_type         = db.Column(db.String(50),  default='manual', index=True)
    source_filename     = db.Column(db.String(255),  nullable=True)

    # Sentiment labels — indexed because every query filters/counts on these
    vader_sentiment     = db.Column(db.String(20), nullable=True)
    textblob_sentiment  = db.Column(db.String(20), nullable=True)
    groq_sentiment      = db.Column(db.String(20), nullable=True)
    groq_explanation    = db.Column(db.Text, nullable=True)
    final_sentiment     = db.Column(db.String(20), nullable=True, index=True)

    # Scores
    vader_compound      = db.Column(db.Float, nullable=True)
    vader_pos           = db.Column(db.Float, nullable=True)
    vader_neg           = db.Column(db.Float, nullable=True)
    vader_neu           = db.Column(db.Float, nullable=True)
    textblob_polarity   = db.Column(db.Float, nullable=True)
    textblob_subjectivity = db.Column(db.Float, nullable=True)

    # NLP stats
    word_count          = db.Column(db.Integer, nullable=True)
    sentence_count      = db.Column(db.Integer, nullable=True)
    top_keywords        = db.Column(db.Text, nullable=True)   # comma-separated

    # Batch reference — indexed for fast batch_result page loads
    batch_id = db.Column(
        db.Integer,
        db.ForeignKey('batch_analyses.id'),
        nullable=True,
        index=True,
    )

    # Indexed because virtually every query sorts or filters by date
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        text = self.input_text or ''
        return {
            'id':                   self.id,
            'input_text':           text[:200] + '…' if len(text) > 200 else text,
            'source_type':          self.source_type,
            'source_filename':      self.source_filename,
            'vader_sentiment':      self.vader_sentiment,
            'textblob_sentiment':   self.textblob_sentiment,
            'groq_sentiment':       self.groq_sentiment,
            'groq_explanation':     self.groq_explanation,
            'final_sentiment':      self.final_sentiment,
            'vader_compound':       self.vader_compound,
            'vader_pos':            self.vader_pos,
            'vader_neg':            self.vader_neg,
            'vader_neu':            self.vader_neu,
            'textblob_polarity':    self.textblob_polarity,
            'textblob_subjectivity':self.textblob_subjectivity,
            'word_count':           self.word_count,
            'sentence_count':       self.sentence_count,
            'top_keywords':         self.top_keywords,
            'batch_id':             self.batch_id,
            'created_at':           self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
        }

    def __repr__(self):
        return f'<AnalysisResult id={self.id} sentiment={self.final_sentiment}>'


class BatchAnalysis(db.Model):
    __tablename__ = 'batch_analyses'

    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    filename       = db.Column(db.String(255), nullable=False)
    file_type      = db.Column(db.String(20),  nullable=False)
    total_records  = db.Column(db.Integer, default=0)
    positive_count = db.Column(db.Integer, default=0)
    negative_count = db.Column(db.Integer, default=0)
    neutral_count  = db.Column(db.Integer, default=0)
    created_at     = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    results = db.relationship('AnalysisResult', backref='batch', lazy='dynamic')
    uploaded_file = db.relationship('UploadedFile', backref='batch', uselist=False)

    def to_dict(self):
        return {
            'id':             self.id,
            'filename':       self.filename,
            'file_type':      self.file_type,
            'total_records':  self.total_records,
            'positive_count': self.positive_count,
            'negative_count': self.negative_count,
            'neutral_count':  self.neutral_count,
            'created_at':     self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
        }

    def __repr__(self):
        return f'<BatchAnalysis id={self.id} file={self.filename}>'


class UploadedFile(db.Model):
    __tablename__ = 'uploaded_files'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    batch_id = db.Column(
        db.Integer,
        db.ForeignKey('batch_analyses.id'),
        nullable=False,
        unique=True,
    )
    filename = db.Column(db.String(255), nullable=False)
    content_type = db.Column(db.String(255), nullable=True)
    size_bytes = db.Column(db.Integer, nullable=False)
    content = db.Column(db.LargeBinary, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self):
        return f'<UploadedFile id={self.id} filename={self.filename}>'
