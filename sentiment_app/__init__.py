"""Sentiment Analysis Application Factory  (v3 — large-data safe)"""

import os
import hmac
import secrets
import tempfile
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from sqlalchemy import text, inspect as sa_inspect
from flask import Flask, abort, g, jsonify, redirect, render_template, request, session, url_for
from sentiment_app.extensions import db, migrate
from sentiment_app.models.database import User

MONGODB_AUTH_SENTINEL = '!managed-in-mongodb!'
APP_SHELL_VERSION = 'sentimentiq-username-nav-v1'


def _ensure_indexes(app):
    """Create DB indexes on existing tables if they don't exist yet.
    SQLite only adds indexes for new tables; ALTER TABLE doesn't support ADD INDEX,
    so we use CREATE INDEX IF NOT EXISTS instead."""
    required = [
        ('ix_analysis_results_final_sentiment', 'analysis_results', 'final_sentiment'),
        ('ix_analysis_results_created_at',      'analysis_results', 'created_at'),
        ('ix_analysis_results_source_type',     'analysis_results', 'source_type'),
        ('ix_analysis_results_batch_id',        'analysis_results', 'batch_id'),
        ('ix_analysis_results_user_id',         'analysis_results', 'user_id'),
        ('ix_batch_analyses_created_at',        'batch_analyses',   'created_at'),
        ('ix_batch_analyses_user_id',           'batch_analyses',   'user_id'),
    ]
    with db.engine.connect() as conn:
        analysis_columns = {
            column['name'] for column in sa_inspect(conn).get_columns('analysis_results')
        }
        batch_columns = {
            column['name'] for column in sa_inspect(conn).get_columns('batch_analyses')
        }
        if 'groq_sentiment' not in analysis_columns:
            conn.execute(text('ALTER TABLE analysis_results ADD COLUMN groq_sentiment VARCHAR(20)'))
        if 'groq_explanation' not in analysis_columns:
            conn.execute(text('ALTER TABLE analysis_results ADD COLUMN groq_explanation TEXT'))
        if 'user_id' not in analysis_columns:
            conn.execute(text(
                'ALTER TABLE analysis_results ADD COLUMN user_id INTEGER REFERENCES users(id)'
            ))
        if 'user_id' not in batch_columns:
            conn.execute(text(
                'ALTER TABLE batch_analyses ADD COLUMN user_id INTEGER REFERENCES users(id)'
            ))
        user_columns = {
            column['name'] for column in sa_inspect(conn).get_columns('users')
        }
        if 'username' not in user_columns:
            conn.execute(text(
                "ALTER TABLE users ADD COLUMN username VARCHAR(50) NOT NULL DEFAULT ''"
            ))
        if 'auth_version' not in user_columns:
            conn.execute(text(
                "ALTER TABLE users ADD COLUMN auth_version INTEGER NOT NULL DEFAULT 0"
            ))
        reset_token_columns = {
            column['name']
            for column in sa_inspect(conn).get_columns('password_reset_tokens')
        }
        if 'failed_attempts' not in reset_token_columns:
            conn.execute(text(
                "ALTER TABLE password_reset_tokens "
                "ADD COLUMN failed_attempts INTEGER NOT NULL DEFAULT 0"
            ))
        if 'token_type' not in reset_token_columns:
            conn.execute(text(
                "ALTER TABLE password_reset_tokens "
                "ADD COLUMN token_type VARCHAR(10) NOT NULL DEFAULT 'email'"
            ))
        for idx_name, table, col in required:
            conn.execute(text(
                f'CREATE INDEX IF NOT EXISTS {idx_name} ON {table} ({col})'
            ))
        conn.commit()


def _migrate_account_credentials(app):
    users_collection = app.extensions['mongo_db']['users']
    with app.app_context():
        for user in User.query.order_by(User.id).all():
            if not user.username:
                user.username = user.email.partition('@')[0][:50] or f'User{user.id}'
            account = users_collection.find_one({'_id': user.id})
            if account is None:
                if user.password_hash == MONGODB_AUTH_SENTINEL:
                    continue
                account = {
                    '_id': user.id,
                    'email': user.email,
                    'email_normalized': user.email.strip().lower(),
                    'password_hash': user.password_hash,
                    'auth_version': user.auth_version,
                    'created_at': user.created_at,
                    'updated_at': user.created_at,
                    'schema_version': 1,
                }
                users_collection.insert_one(account)
            elif account.get('email_normalized') != user.email.strip().lower():
                raise RuntimeError(
                    f'MongoDB account id {user.id} does not match its SQLite account email.'
                )

            mongo_auth_version = account.get('auth_version', user.auth_version)
            if user.password_hash != MONGODB_AUTH_SENTINEL:
                user.password_hash = MONGODB_AUTH_SENTINEL
            if user.auth_version != mongo_auth_version:
                user.auth_version = mongo_auth_version

        db.session.commit()


def create_app():
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
    matplotlib_config = os.path.join(tempfile.gettempdir(), 'sentimentiq-matplotlib')
    os.makedirs(matplotlib_config, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', matplotlib_config)
    app = Flask(__name__)

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    app.config['SECRET_KEY']                  = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
    app.config['SQLALCHEMY_DATABASE_URI']     = os.environ.get(
        'DATABASE_URL',
        'sqlite:///' + os.path.join(BASE_DIR, '..', 'sentiment.db'),
    )
    app.config['MONGODB_URI']                 = os.environ.get(
        'MONGODB_URI',
        'mongodb://localhost:27017',
    )
    app.config['MONGODB_DATABASE']            = os.environ.get(
        'MONGODB_DATABASE',
        'sentimentiq',
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['UPLOAD_FOLDER']               = os.path.join(BASE_DIR, 'uploads')
    app.config['MAX_CONTENT_LENGTH']          = 32 * 1024 * 1024
    app.config['SESSION_COOKIE_HTTPONLY']     = True
    app.config['SESSION_COOKIE_SAMESITE']     = 'Lax'
    app.config['SESSION_COOKIE_SECURE']        = (
        os.environ.get('SESSION_COOKIE_SECURE', 'false').lower() == 'true'
    )
    app.config['MAIL_SERVER']                 = os.environ.get('MAIL_SERVER')
    app.config['MAIL_PORT']                   = int(os.environ.get('MAIL_PORT', '587'))
    app.config['MAIL_USE_TLS']                = os.environ.get('MAIL_USE_TLS', 'true').lower() == 'true'
    app.config['MAIL_USE_SSL']                = os.environ.get('MAIL_USE_SSL', 'false').lower() == 'true'
    app.config['MAIL_USERNAME']               = os.environ.get('MAIL_USERNAME')
    app.config['MAIL_PASSWORD']               = os.environ.get('MAIL_PASSWORD')
    app.config['MAIL_DEFAULT_SENDER']         = os.environ.get('MAIL_DEFAULT_SENDER')
    app.config['PUBLIC_BASE_URL']             = os.environ.get('PUBLIC_BASE_URL', '').rstrip('/')
    app.config['PASSWORD_RESET_CONSOLE_OTP']  = (
        os.environ.get('PASSWORD_RESET_CONSOLE_OTP', 'false').lower() == 'true'
    )
    app.config['GROQ_API_KEY']                = os.environ.get('GROQ_API_KEY', '').strip()
    app.config['GROQ_MODEL']                  = os.environ.get(
        'GROQ_MODEL',
        'llama-3.3-70b-versatile',
    ).strip() or 'llama-3.3-70b-versatile'

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    db.init_app(app)
    migrate.init_app(app, db)
    mongo_client = MongoClient(app.config['MONGODB_URI'])
    try:
        mongo_client.admin.command('ping')
        mongo_database = mongo_client[app.config['MONGODB_DATABASE']]
        mongo_database['users'].create_index('email_normalized', unique=True)
    except PyMongoError:
        mongo_client.close()
        raise
    app.extensions['mongo_client'] = mongo_client
    app.extensions['mongo_db'] = mongo_database

    from sentiment_app.routes import bp
    from sentiment_app.auth import auth_bp
    app.register_blueprint(bp)
    app.register_blueprint(auth_bp)

    @app.before_request
    def load_user_and_protect_routes():
        user_id = session.get('user_id')
        g.current_user = db.session.get(User, user_id) if user_id else None
        g.mongo_account = (
            app.extensions['mongo_db']['users'].find_one({'_id': user_id})
            if g.current_user is not None
            else None
        )

        if (
            g.current_user is not None
            and (
                g.mongo_account is None
                or session.get('auth_version') != g.mongo_account.get('auth_version')
            )
        ):
            session.clear()
            g.current_user = None
            g.mongo_account = None

        if g.current_user is None:
            session.pop('user_id', None)
            if request.endpoint not in {
                'auth.login',
                'auth.register',
                'auth.forgot_password',
                'auth.reset_password',
                'auth.reset_password_otp',
                'healthz',
                'static',
            }:
                if request.path.startswith('/api/'):
                    return jsonify({'error': 'Authentication required.'}), 401
                return redirect(url_for('auth.login', next=request.full_path))

        if request.method == 'POST':
            expected = session.get('_csrf_token')
            provided = request.form.get('csrf_token') or request.headers.get('X-CSRF-Token')
            if (
                not isinstance(expected, str)
                or not isinstance(provided, str)
                or not hmac.compare_digest(expected, provided)
            ):
                if request.path.startswith('/api/'):
                    return jsonify({'error': 'Invalid CSRF token.'}), 400
                abort(400, description='Invalid CSRF token.')

    @app.get('/healthz')
    def healthz():
        return jsonify({'status': 'ok'})

    @app.context_processor
    def inject_auth_context():
        def csrf_token():
            token = session.get('_csrf_token')
            if token is None:
                token = secrets.token_urlsafe(32)
                session['_csrf_token'] = token
            return token

        return {
            'current_user': g.get('current_user'),
            'csrf_token': csrf_token,
            'app_shell_version': APP_SHELL_VERSION,
        }

    @app.after_request
    def prevent_password_reset_token_leaks(response):
        if request.endpoint in {'auth.reset_password', 'auth.reset_password_otp'}:
            response.headers['Referrer-Policy'] = 'no-referrer'
            response.headers['Cache-Control'] = 'no-store'
        return response

    with app.app_context():
        db.create_all()
        # Ensure indexes exist on pre-existing databases (SQLite doesn't auto-add them)
        _ensure_indexes(app)
        _migrate_account_credentials(app)

    # ── Error handlers ────────────────────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return render_template('errors/404.html'), 404

    @app.errorhandler(500)
    def server_error(e):
        return render_template('errors/500.html'), 500

    @app.errorhandler(413)
    def too_large(e):
        return render_template('errors/413.html'), 413

    return app
