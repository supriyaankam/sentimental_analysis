"""Account creation, login, and password recovery routes."""

import hashlib
import hmac
import logging
import secrets
import smtplib
import ssl
from datetime import datetime, timedelta
from email.message import EmailMessage
from urllib.parse import urlsplit

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from pymongo.errors import DuplicateKeyError, PyMongoError
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from werkzeug.security import check_password_hash, generate_password_hash

from sentiment_app.extensions import db
from sentiment_app.models.database import PasswordResetToken, User
from sentiment_app import MONGODB_AUTH_SENTINEL

auth_bp = Blueprint('auth', __name__)
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 128
PASSWORD_RESET_TTL = timedelta(minutes=30)
PASSWORD_RESET_RATE_WINDOW = timedelta(minutes=15)
MAX_RESET_EMAILS_PER_WINDOW = 3
MAX_RESET_OTP_ATTEMPTS = 5
logger = logging.getLogger(__name__)


def _password_policy_error(password):
    if not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
        return f'Password must be {MIN_PASSWORD_LENGTH}-{MAX_PASSWORD_LENGTH} characters long.'
    if not any(char.isupper() for char in password):
        return 'Password must contain at least one uppercase letter.'
    if not any(char.islower() for char in password):
        return 'Password must contain at least one lowercase letter.'
    if not any(char.isdigit() for char in password):
        return 'Password must contain at least one number.'
    if not any(not char.isalnum() for char in password):
        return 'Password must contain at least one special character.'
    return None


def _password_matches(mongo_account, password):
    return (
        mongo_account is not None
        and len(password) <= MAX_PASSWORD_LENGTH
        and check_password_hash(
            mongo_account.get('password_hash', ''),
            password,
        )
    )


def _users_collection():
    return current_app.extensions['mongo_db']['users']


def _safe_next_url(target):
    if not target or '\\' in target:
        return None
    parts = urlsplit(target)
    if parts.scheme or parts.netloc or not parts.path.startswith('/') or parts.path.startswith('//'):
        return None
    return target


def _hash_reset_token(token):
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def _hash_reset_otp(user_id, otp):
    secret = current_app.config['SECRET_KEY'].encode('utf-8')
    message = f'{user_id}:{otp}'.encode('utf-8')
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def _console_otp_enabled():
    return current_app.debug and current_app.config['PASSWORD_RESET_CONSOLE_OTP']


def _send_password_reset_email(user, reset_url):
    mail_server = current_app.config['MAIL_SERVER']
    sender = current_app.config['MAIL_DEFAULT_SENDER']
    public_base_url = current_app.config['PUBLIC_BASE_URL']
    if not mail_server or not sender or not public_base_url:
        raise RuntimeError(
            'Password reset email is not configured. '
            'Set MAIL_SERVER, MAIL_DEFAULT_SENDER, and PUBLIC_BASE_URL.'
        )

    message = EmailMessage()
    message['Subject'] = 'Reset your SentimentIQ password'
    message['From'] = sender
    message['To'] = user.email
    message.set_content(
        'A password reset was requested for your SentimentIQ account.\n\n'
        f'Use this link within 30 minutes to choose a new password:\n{reset_url}\n\n'
        'If you did not request this, you can ignore this email.'
    )

    host = current_app.config['MAIL_SERVER']
    port = current_app.config['MAIL_PORT']
    username = current_app.config['MAIL_USERNAME']
    password = current_app.config['MAIL_PASSWORD']
    timeout = 15

    if current_app.config['MAIL_USE_SSL']:
        with smtplib.SMTP_SSL(host, port, timeout=timeout, context=ssl.create_default_context()) as smtp:
            if username:
                smtp.login(username, password or '')
            smtp.send_message(message)
        return

    with smtplib.SMTP(host, port, timeout=timeout) as smtp:
        if current_app.config['MAIL_USE_TLS']:
            smtp.starttls(context=ssl.create_default_context())
        if username:
            smtp.login(username, password or '')
        smtp.send_message(message)


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        password_confirm = request.form.get('password_confirm', '')

        password_error = _password_policy_error(password)
        if not username or len(username) > 50:
            flash('Enter a username up to 50 characters long.', 'danger')
        elif len(email) > 254 or '@' not in email or any(char.isspace() for char in email):
            flash('Enter a valid email address.', 'danger')
        elif password_error:
            flash(password_error, 'danger')
        elif password != password_confirm:
            flash('The passwords do not match.', 'danger')
        else:
            user = User.query.filter_by(email=email).first()
            existing_mongo_account = _users_collection().find_one({
                'email_normalized': email,
            })
            if existing_mongo_account is not None:
                flash('An account with that email address already exists.', 'danger')
                return render_template(
                    'auth/register.html',
                    username=username,
                    email=email,
                )

            if user is None:
                user = User(
                    username=username,
                    email=email,
                    password_hash=MONGODB_AUTH_SENTINEL,
                )
                db.session.add(user)
                auth_version = 0
            else:
                account_with_id = _users_collection().find_one({'_id': user.id})
                if account_with_id is not None:
                    logger.error(
                        'Cannot restore account %s: its MongoDB id is assigned to '
                        'a different email address.',
                        user.id,
                    )
                    flash('The account could not be created. Please contact support.', 'danger')
                    return render_template(
                        'auth/register.html',
                        username=username,
                        email=email,
                    )
                auth_version = user.auth_version + 1

            user.username = username
            user.password_hash = MONGODB_AUTH_SENTINEL
            user.auth_version = auth_version
            user_id = None
            mongo_inserted = False
            try:
                db.session.flush()
                user_id = user.id
                _users_collection().insert_one({
                    '_id': user_id,
                    'email': email,
                    'email_normalized': email,
                    'password_hash': generate_password_hash(password),
                    'auth_version': auth_version,
                    'created_at': user.created_at,
                    'updated_at': datetime.utcnow(),
                    'schema_version': 1,
                })
                mongo_inserted = True
                db.session.commit()
            except IntegrityError:
                db.session.rollback()
                flash('An account with that email address already exists.', 'danger')
            except DuplicateKeyError:
                db.session.rollback()
                flash('An account with that email address already exists.', 'danger')
            except (PyMongoError, SQLAlchemyError):
                db.session.rollback()
                if mongo_inserted:
                    _users_collection().delete_one({'_id': user_id})
                logger.exception('Could not create account in the configured databases.')
                flash('The account could not be created. Please try again later.', 'danger')
            else:
                session.clear()
                session['user_id'] = user.id
                session['auth_version'] = auth_version
                flash('Your account has been created.', 'success')
                return redirect(url_for('main.index'))

        return render_template(
            'auth/register.html',
            username=username,
            email=email,
        )

    return render_template('auth/register.html', username='', email='')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    next_url = _safe_next_url(request.values.get('next'))
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user = User.query.filter_by(email=email).first()
        mongo_account = _users_collection().find_one({
            'email_normalized': email,
        })

        if user is None or mongo_account is None or mongo_account.get('_id') != user.id:
            flash('Invalid email or password.', 'danger')
            return render_template('auth/login.html', email=email, next_url=next_url)

        if not _password_matches(mongo_account, password):
            flash('Invalid email or password.', 'danger')
            return render_template('auth/login.html', email=email, next_url=next_url)

        session.clear()
        session['user_id'] = user.id
        session['auth_version'] = mongo_account.get('auth_version', 0)
        flash('You are now logged in.', 'success')
        return redirect(next_url or url_for('main.index'))

    return render_template('auth/login.html', email='', next_url=next_url)


@auth_bp.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        user = User.query.filter_by(email=email).first() if len(email) <= 254 else None

        if user:
            now = datetime.utcnow()
            PasswordResetToken.query.filter(
                PasswordResetToken.expires_at <= now,
            ).delete(synchronize_session=False)
            recent_requests = PasswordResetToken.query.filter(
                PasswordResetToken.user_id == user.id,
                PasswordResetToken.created_at >= now - PASSWORD_RESET_RATE_WINDOW,
            ).count()

            if recent_requests < MAX_RESET_EMAILS_PER_WINDOW:
                use_console_otp = _console_otp_enabled()
                raw_token = (
                    f'{secrets.randbelow(1_000_000):06d}'
                    if use_console_otp
                    else secrets.token_urlsafe(32)
                )
                reset_record = PasswordResetToken(
                    user_id=user.id,
                    token_hash=(
                        _hash_reset_otp(user.id, raw_token)
                        if use_console_otp
                        else _hash_reset_token(raw_token)
                    ),
                    token_type='otp' if use_console_otp else 'email',
                    expires_at=now + PASSWORD_RESET_TTL,
                )
                db.session.add(reset_record)
                db.session.commit()

                if use_console_otp:
                    logger.warning(
                        'LOCAL DEVELOPMENT PASSWORD RESET OTP: %s '
                        '(expires in 30 minutes; do not use this mode in production)',
                        raw_token,
                    )
                else:
                    public_base_url = current_app.config['PUBLIC_BASE_URL']
                    if public_base_url:
                        reset_url = f"{public_base_url}{url_for('auth.reset_password', token=raw_token)}"
                        try:
                            _send_password_reset_email(user, reset_url)
                        except (OSError, smtplib.SMTPException, RuntimeError):
                            logger.exception('Could not send password reset email.')
                            db.session.delete(reset_record)
                            db.session.commit()
                    else:
                        logger.error(
                            'Could not send password reset email: PUBLIC_BASE_URL is not configured.'
                        )
                        db.session.delete(reset_record)
                        db.session.commit()

        flash(
            'If an account exists for that email, password reset instructions are available.',
            'info',
        )
        if _console_otp_enabled():
            return redirect(url_for('auth.reset_password_otp'))
        return redirect(url_for('auth.login'))

    return render_template(
        'auth/forgot_password.html',
        console_otp_enabled=_console_otp_enabled(),
    )


@auth_bp.route('/reset-password-otp', methods=['GET', 'POST'])
def reset_password_otp():
    if not _console_otp_enabled():
        abort(404)

    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        otp = request.form.get('otp', '').strip()
        new_password = request.form.get('new_password', '')
        password_confirm = request.form.get('password_confirm', '')
        password_error = _password_policy_error(new_password)
        user = User.query.filter_by(email=email).first() if len(email) <= 254 else None
        now = datetime.utcnow()
        reset_record = None

        if user is not None:
            reset_record = PasswordResetToken.query.filter(
                PasswordResetToken.user_id == user.id,
                PasswordResetToken.token_type == 'otp',
                PasswordResetToken.expires_at > now,
            ).order_by(PasswordResetToken.created_at.desc()).first()

        if (
            user is None
            or reset_record is None
            or reset_record.failed_attempts >= MAX_RESET_OTP_ATTEMPTS
            or not hmac.compare_digest(
                reset_record.token_hash,
                _hash_reset_otp(user.id, otp),
            )
        ):
            if (
                reset_record is not None
                and reset_record.failed_attempts < MAX_RESET_OTP_ATTEMPTS
            ):
                PasswordResetToken.query.filter_by(
                    id=reset_record.id,
                ).filter(
                    PasswordResetToken.failed_attempts < MAX_RESET_OTP_ATTEMPTS,
                    PasswordResetToken.expires_at > datetime.utcnow(),
                ).update(
                    {
                        'failed_attempts': PasswordResetToken.failed_attempts + 1,
                    },
                    synchronize_session=False,
                )
                db.session.commit()
            flash('The email or OTP is invalid, expired, or already used.', 'danger')
            return render_template('auth/reset_password_otp.html', email=email)

        if password_error:
            flash(password_error, 'danger')
            return render_template('auth/reset_password_otp.html', email=email)
        if new_password != password_confirm:
            flash('The passwords do not match.', 'danger')
            return render_template('auth/reset_password_otp.html', email=email)

        consumed = PasswordResetToken.query.filter_by(
            id=reset_record.id,
            token_hash=reset_record.token_hash,
            token_type='otp',
            failed_attempts=reset_record.failed_attempts,
        ).filter(
            PasswordResetToken.expires_at > datetime.utcnow(),
        ).delete(synchronize_session=False)
        if consumed != 1:
            db.session.rollback()
            flash('The email or OTP is invalid, expired, or already used.', 'danger')
            return render_template('auth/reset_password_otp.html', email=email)
        db.session.commit()

        account = _users_collection().find_one({'_id': user.id})
        if account is None:
            flash('The email or OTP is invalid, expired, or already used.', 'danger')
            return render_template('auth/reset_password_otp.html', email=email)

        old_auth_version = account.get('auth_version', 0)
        updated = _users_collection().update_one(
            {'_id': user.id, 'auth_version': old_auth_version},
            {
                '$set': {
                    'password_hash': generate_password_hash(new_password),
                    'updated_at': datetime.utcnow(),
                },
                '$inc': {'auth_version': 1},
            },
        )
        if updated.matched_count != 1:
            flash('Your account changed. Request a new reset OTP.', 'danger')
            return render_template('auth/reset_password_otp.html', email=email)

        user.password_hash = MONGODB_AUTH_SENTINEL
        user.auth_version = old_auth_version + 1
        PasswordResetToken.query.filter_by(user_id=user.id).delete()
        db.session.commit()
        session.clear()
        session['user_id'] = user.id
        session['auth_version'] = old_auth_version + 1
        flash('Your password has been reset. Other active sessions have been signed out.', 'success')
        return redirect(url_for('main.index'))

    return render_template('auth/reset_password_otp.html', email='')


@auth_bp.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    token_record = PasswordResetToken.query.filter_by(
        token_hash=_hash_reset_token(token),
    ).first()
    if token_record is None:
        flash('This password reset link is invalid or has expired. Request a new one.', 'danger')
        return redirect(url_for('auth.forgot_password'))

    if token_record.expires_at <= datetime.utcnow():
        db.session.delete(token_record)
        db.session.commit()
        flash('This password reset link is invalid or has expired. Request a new one.', 'danger')
        return redirect(url_for('auth.forgot_password'))

    if request.method == 'POST':
        new_password = request.form.get('new_password', '')
        password_confirm = request.form.get('password_confirm', '')
        password_error = _password_policy_error(new_password)

        if password_error:
            flash(password_error, 'danger')
        elif password_confirm != new_password:
            flash('The passwords do not match.', 'danger')
        else:
            user = token_record.user
            account = _users_collection().find_one({'_id': user.id})
            if account is None:
                flash('This password reset link is invalid or has expired. Request a new one.', 'danger')
                return redirect(url_for('auth.forgot_password'))

            old_auth_version = account.get('auth_version', 0)
            updated = _users_collection().update_one(
                {'_id': user.id, 'auth_version': old_auth_version},
                {
                    '$set': {
                        'password_hash': generate_password_hash(new_password),
                        'updated_at': datetime.utcnow(),
                    },
                    '$inc': {'auth_version': 1},
                },
            )
            if updated.matched_count != 1:
                db.session.rollback()
                flash('This password reset link is invalid or has expired. Request a new one.', 'danger')
                return redirect(url_for('auth.forgot_password'))

            consumed = PasswordResetToken.query.filter_by(
                id=token_record.id,
                token_hash=_hash_reset_token(token),
            ).filter(
                PasswordResetToken.expires_at > datetime.utcnow(),
            ).delete(synchronize_session=False)
            if consumed != 1:
                db.session.rollback()
                flash('This password reset link is invalid or has expired. Request a new one.', 'danger')
                return redirect(url_for('auth.forgot_password'))

            user.password_hash = MONGODB_AUTH_SENTINEL
            user.auth_version = old_auth_version + 1
            PasswordResetToken.query.filter_by(user_id=user.id).delete()
            db.session.commit()
            session.clear()
            session['user_id'] = user.id
            session['auth_version'] = old_auth_version + 1
            flash('Your password has been reset. Other active sessions have been signed out.', 'success')
            return redirect(url_for('main.index'))

    return render_template('auth/reset_password.html', token=token)


@auth_bp.route('/change-password', methods=['GET', 'POST'])
def change_password():
    user = g.current_user
    if request.method == 'POST':
        current_password = request.form.get('current_password', '')
        new_password = request.form.get('new_password', '')
        password_confirm = request.form.get('password_confirm', '')
        password_error = _password_policy_error(new_password)

        if not _password_matches(g.mongo_account, current_password):
            flash('The current password is incorrect.', 'danger')
        elif password_error:
            flash(password_error, 'danger')
        elif _password_matches(g.mongo_account, new_password):
            flash('Choose a new password different from your current password.', 'danger')
        elif new_password != password_confirm:
            flash('The new passwords do not match.', 'danger')
        else:
            account = g.mongo_account
            old_auth_version = account.get('auth_version', 0)
            updated = _users_collection().update_one(
                {'_id': user.id, 'auth_version': old_auth_version},
                {
                    '$set': {
                        'password_hash': generate_password_hash(new_password),
                        'updated_at': datetime.utcnow(),
                    },
                    '$inc': {'auth_version': 1},
                },
            )
            if updated.matched_count != 1:
                session.clear()
                flash('Your account changed in another session. Please log in again.', 'warning')
                return redirect(url_for('auth.login'))

            user.password_hash = MONGODB_AUTH_SENTINEL
            user.auth_version = old_auth_version + 1
            db.session.commit()
            session.clear()
            session['user_id'] = user.id
            session['auth_version'] = old_auth_version + 1
            flash('Your password has been changed. Other active sessions have been signed out.', 'success')
            return redirect(url_for('main.index'))

        return render_template('auth/change_password.html')

    return render_template('auth/change_password.html')


@auth_bp.route('/logout', methods=['POST'])
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))
