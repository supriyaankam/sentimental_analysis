import io
import hashlib
import hmac
import os
import re
import sqlite3
import smtplib
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from groq import GroqError
from sqlalchemy import inspect
from werkzeug.security import check_password_hash

from sentiment_app import create_app
from sentiment_app.extensions import db
from sentiment_app.models.database import (
    AnalysisResult,
    PasswordResetToken,
    UploadedFile,
    User,
)
from sentiment_app.groq_engine import analyze_with_groq


class AuthenticationAndUploadTests(unittest.TestCase):
    PASSWORD = 'Correct-Horse-Battery9!'

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.previous_database_url = os.environ.get('DATABASE_URL')
        cls.previous_secret_key = os.environ.get('SECRET_KEY')
        cls.previous_mongodb_database = os.environ.get('MONGODB_DATABASE')
        os.environ['DATABASE_URL'] = f"sqlite:///{os.path.join(cls.temp_dir.name, 'test.db')}"
        os.environ['SECRET_KEY'] = 'test-secret-key'
        os.environ['MONGODB_DATABASE'] = f"sentimentiq_test_{uuid.uuid4().hex}"
        cls.app = create_app()
        cls.app.config['TESTING'] = True
        cls.app.config.update(
            DEBUG=False,
            PASSWORD_RESET_CONSOLE_OTP=False,
            MAIL_SERVER='smtp.test',
            MAIL_PORT=587,
            MAIL_USE_TLS=True,
            MAIL_USE_SSL=False,
            MAIL_USERNAME='mailer',
            MAIL_PASSWORD='mailer-password',
            MAIL_DEFAULT_SENDER='SentimentIQ <no-reply@example.com>',
            PUBLIC_BASE_URL='https://sentiment.example.com',
            GROQ_API_KEY='',
            GROQ_MODEL='llama-3.3-70b-versatile',
        )

    @classmethod
    def tearDownClass(cls):
        with cls.app.app_context():
            db.session.remove()
            db.engine.dispose()
        cls.app.extensions['mongo_client'].drop_database(
            cls.app.config['MONGODB_DATABASE']
        )
        cls.app.extensions['mongo_client'].close()
        if cls.previous_database_url is None:
            os.environ.pop('DATABASE_URL', None)
        else:
            os.environ['DATABASE_URL'] = cls.previous_database_url
        if cls.previous_secret_key is None:
            os.environ.pop('SECRET_KEY', None)
        else:
            os.environ['SECRET_KEY'] = cls.previous_secret_key
        if cls.previous_mongodb_database is None:
            os.environ.pop('MONGODB_DATABASE', None)
        else:
            os.environ['MONGODB_DATABASE'] = cls.previous_mongodb_database
        cls.temp_dir.cleanup()

    def setUp(self):
        self.app.config['DEBUG'] = False
        self.app.config['PASSWORD_RESET_CONSOLE_OTP'] = False
        with self.app.app_context():
            db.drop_all()
            db.create_all()
        self.app.extensions['mongo_db']['users'].delete_many({})
        self.client = self.app.test_client()

    def _csrf_token(self, path):
        page = self.client.get(path)
        self.assertEqual(page.status_code, 200)
        match = re.search(
            r'name="csrf_token" value="([^"]+)"',
            page.get_data(as_text=True),
        )
        self.assertIsNotNone(match)
        return match.group(1)

    def _register(self, email='user@example.com', username='Test User'):
        token = self._csrf_token('/register')
        return self.client.post('/register', data={
            'csrf_token': token,
            'username': username,
            'email': email,
            'password': self.PASSWORD,
            'password_confirm': self.PASSWORD,
        })

    def test_registration_hashes_password_and_requires_login(self):
        self.assertEqual(self.client.get('/').status_code, 302)
        response = self._register()
        self.assertEqual(response.status_code, 302)

        with self.app.app_context():
            user = User.query.filter_by(email='user@example.com').one()
            account = self.app.extensions['mongo_db']['users'].find_one({
                'email_normalized': 'user@example.com',
            })
            self.assertEqual(user.username, 'Test User')
            self.assertEqual(user.password_hash, '!managed-in-mongodb!')
            self.assertIsNotNone(account)
            self.assertNotEqual(account['password_hash'], self.PASSWORD)
            self.assertTrue(check_password_hash(account['password_hash'], self.PASSWORD))
            self.assertEqual(account['email'], 'user@example.com')
            self.assertEqual(account['auth_version'], 0)

        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertEqual(self.client.get('/api/dashboard').status_code, 200)

    def test_authenticated_home_shows_logout_without_change_password_link(self):
        self._register()

        response = self.client.get('/')
        page = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn('Logout', page)
        self.assertIn('action="/logout"', page)
        self.assertIn('class="navbar-text account-name small me-2">Test User</span>', page)
        self.assertIn('data-app-shell="sentimentiq-username-nav-v1"', page)
        self.assertNotIn('user@example.com', page)
        self.assertNotIn('Change password', page)

    def test_registration_requires_username(self):
        token = self._csrf_token('/register')
        response = self.client.post('/register', data={
            'csrf_token': token,
            'email': 'missing-username@example.com',
            'password': self.PASSWORD,
            'password_confirm': self.PASSWORD,
        })

        self.assertEqual(response.status_code, 200)
        self.assertIn('name="username"', response.get_data(as_text=True))
        self.assertIn('Enter a username', response.get_data(as_text=True))
        with self.app.app_context():
            self.assertIsNone(
                User.query.filter_by(email='missing-username@example.com').first()
            )

    def test_health_check_is_public(self):
        response = self.client.get('/healthz')

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {'status': 'ok'})

    def test_registration_recovers_sqlite_account_missing_from_mongodb(self):
        self._register()
        with self.app.app_context():
            user = User.query.filter_by(email='user@example.com').one()
            original_user_id = user.id
            db.session.add(AnalysisResult(
                user_id=user.id,
                input_text='Existing account analysis',
                final_sentiment='positive',
            ))
            db.session.commit()
            self.app.extensions['mongo_db']['users'].delete_one({'_id': user.id})

        self.client.post('/logout', data={'csrf_token': self._csrf_token('/login')})
        response = self._register()
        self.assertEqual(response.status_code, 302)

        with self.app.app_context():
            user = User.query.filter_by(email='user@example.com').one()
            account = self.app.extensions['mongo_db']['users'].find_one({
                'email_normalized': 'user@example.com',
            })
            self.assertEqual(user.id, original_user_id)
            self.assertEqual(user.auth_version, 1)
            self.assertEqual(account['_id'], original_user_id)
            self.assertEqual(account['auth_version'], 1)
            self.assertTrue(check_password_hash(account['password_hash'], self.PASSWORD))
            self.assertEqual(
                AnalysisResult.query.filter_by(user_id=original_user_id).count(),
                1,
            )

    def test_startup_credential_migration_does_not_restore_deleted_mongodb_user(self):
        self._register()
        with self.app.app_context():
            user = User.query.filter_by(email='user@example.com').one()
            user_id = user.id
            self.app.extensions['mongo_db']['users'].delete_one({'_id': user_id})

        from sentiment_app import _migrate_account_credentials

        _migrate_account_credentials(self.app)
        self.assertIsNone(
            self.app.extensions['mongo_db']['users'].find_one({'_id': user_id})
        )

    def test_login_accepts_valid_password_and_rejects_invalid_password(self):
        self._register()
        self.client.post('/logout', data={'csrf_token': self._csrf_token('/')})

        token = self._csrf_token('/login')
        rejected = self.client.post('/login', data={
            'csrf_token': token,
            'email': 'user@example.com',
            'password': 'incorrect-password',
        })
        self.assertEqual(rejected.status_code, 200)

        accepted = self.client.post('/login', data={
            'csrf_token': token,
            'email': 'USER@example.com',
            'password': self.PASSWORD,
        })
        self.assertEqual(accepted.status_code, 302)
        self.assertEqual(self.client.get('/').status_code, 200)
        with self.app.app_context():
            account = self.app.extensions['mongo_db']['users'].find_one({
                'email_normalized': 'user@example.com',
            })
            self.assertTrue(check_password_hash(account['password_hash'], self.PASSWORD))

    def test_login_page_does_not_show_forgot_password_option(self):
        response = self.client.get('/login')
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('Forgot password?', response.get_data(as_text=True))

    def test_dashboard_trend_aggregates_sentiments_per_day(self):
        self._register()
        with self.app.app_context():
            user = User.query.filter_by(email='user@example.com').one()
            db.session.add_all([
                AnalysisResult(
                    user_id=user.id,
                    input_text='Positive sample',
                    final_sentiment='positive',
                    created_at=datetime(2026, 10, 20, 9),
                ),
                AnalysisResult(
                    user_id=user.id,
                    input_text='Another positive sample',
                    final_sentiment='positive',
                    created_at=datetime(2026, 10, 20, 15),
                ),
                AnalysisResult(
                    user_id=user.id,
                    input_text='Negative sample',
                    final_sentiment='negative',
                    created_at=datetime(2026, 10, 21, 10),
                ),
            ])
            db.session.commit()

        response = self.client.get('/api/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['trend'], {
            'dates': ['2026-10-20', '2026-10-21'],
            'positive': [2, 0],
            'negative': [0, 1],
            'neutral': [0, 0],
            'mixed': [0, 0],
        })

    def test_dashboard_trend_returns_one_day_for_single_day_analyses(self):
        self._register()
        with self.app.app_context():
            user = User.query.filter_by(email='user@example.com').one()
            db.session.add_all([
                AnalysisResult(
                    user_id=user.id,
                    input_text=f'Sample {index}',
                    final_sentiment=sentiment,
                    created_at=datetime(2026, 10, 21, 10, index),
                )
                for index, sentiment in enumerate(
                    ('positive', 'negative', 'neutral', 'mixed'),
                )
            ])
            db.session.commit()

        response = self.client.get('/api/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['trend'], {
            'dates': ['2026-10-21'],
            'positive': [1],
            'negative': [1],
            'neutral': [1],
            'mixed': [1],
        })

    def test_groq_uses_configured_key_and_returns_valid_json_sentiment(self):
        self.app.config.update(
            GROQ_API_KEY='test-groq-key',
            GROQ_MODEL='test-model',
        )
        completion = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content='{"sentiment":"positive","explanation":"The text is upbeat."}',
                ),
            )],
        )
        with patch('sentiment_app.groq_engine.Groq') as groq_client:
            groq_client.return_value.chat.completions.create.return_value = completion
            with self.app.app_context():
                result = analyze_with_groq('A positive example.')

        self.assertEqual(result, {
            'sentiment': 'positive',
            'explanation': 'The text is upbeat.',
        })
        groq_client.assert_called_once_with(api_key='test-groq-key')
        request_args = groq_client.return_value.chat.completions.create.call_args
        self.assertEqual(request_args.kwargs['model'], 'test-model')
        self.assertEqual(request_args.kwargs['response_format'], {'type': 'json_object'})
        self.assertEqual(request_args.kwargs['timeout'], 20.0)

    def test_groq_without_configured_key_is_skipped(self):
        self.app.config['GROQ_API_KEY'] = ''
        with patch('sentiment_app.groq_engine.Groq') as groq_client:
            with self.app.app_context():
                result = analyze_with_groq('A test.')

        self.assertIsNone(result)
        groq_client.assert_not_called()

    def test_groq_api_errors_are_logged_and_reported_without_exposing_details(self):
        self.app.config['GROQ_API_KEY'] = 'test-groq-key'
        with patch(
            'sentiment_app.groq_engine.Groq',
            side_effect=GroqError('private client diagnostic'),
        ):
            with self.assertLogs('sentiment_app.groq_engine', level='ERROR'):
                with self.app.app_context():
                    result = analyze_with_groq('A test.')

        self.assertEqual(result, {
            'sentiment': None,
            'explanation': 'Groq analysis failed. Check the server log.',
        })

    def test_registration_rejects_password_missing_required_character_classes(self):
        token = self._csrf_token('/register')
        response = self.client.post('/register', data={
            'csrf_token': token,
            'username': 'Weak User',
            'email': 'weak@example.com',
            'password': 'alllowercase123',
            'password_confirm': 'alllowercase123',
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(
            'at least one uppercase letter',
            response.get_data(as_text=True),
        )
        with self.app.app_context():
            self.assertIsNone(User.query.filter_by(email='weak@example.com').first())

    def test_change_password_checks_current_password_and_revokes_other_sessions(self):
        self._register()
        other_client = self.app.test_client()
        other_login_page = other_client.get('/login').get_data(as_text=True)
        login_token = re.search(
            r'name="csrf_token" value="([^"]+)"',
            other_login_page,
        ).group(1)
        other_client.post('/login', data={
            'csrf_token': login_token,
            'email': 'user@example.com',
            'password': self.PASSWORD,
        })
        change_token = self._csrf_token('/change-password')
        rejected = self.client.post('/change-password', data={
            'csrf_token': change_token,
            'current_password': 'incorrect-current-password',
            'new_password': 'New-Strong-Password5!',
            'password_confirm': 'New-Strong-Password5!',
        })
        self.assertEqual(rejected.status_code, 200)
        self.assertIn('current password is incorrect', rejected.get_data(as_text=True))

        changed = self.client.post('/change-password', data={
            'csrf_token': change_token,
            'current_password': self.PASSWORD,
            'new_password': 'New-Strong-Password5!',
            'password_confirm': 'New-Strong-Password5!',
        })
        self.assertEqual(changed.status_code, 302)
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertEqual(other_client.get('/').status_code, 302)
        with self.app.app_context():
            account = self.app.extensions['mongo_db']['users'].find_one({
                'email_normalized': 'user@example.com',
            })
            self.assertTrue(check_password_hash(
                account['password_hash'],
                'New-Strong-Password5!',
            ))
            self.assertEqual(account['auth_version'], 1)

        self.client.post('/logout', data={'csrf_token': self._csrf_token('/')})
        token = self._csrf_token('/login')
        old_password_login = self.client.post('/login', data={
            'csrf_token': token,
            'email': 'user@example.com',
            'password': self.PASSWORD,
        })
        self.assertEqual(old_password_login.status_code, 200)
        new_password_login = self.client.post('/login', data={
            'csrf_token': token,
            'email': 'user@example.com',
            'password': 'New-Strong-Password5!',
        })
        self.assertEqual(new_password_login.status_code, 302)

    def test_post_without_csrf_token_is_rejected(self):
        self._register()
        response = self.client.post('/analyze', data={'text': 'Nice'})
        self.assertEqual(response.status_code, 400)

    def test_forgot_password_sends_reset_link_and_token_is_single_use(self):
        self._register()
        token = self._csrf_token('/forgot-password')
        sent_messages = []
        with patch(
            'sentiment_app.auth._send_password_reset_email',
            side_effect=lambda user, reset_url: sent_messages.append((user.email, reset_url)),
        ) as send_email:
            response = self.client.post('/forgot-password', data={
                'csrf_token': token,
                'email': 'USER@example.com',
            })
        self.assertEqual(response.status_code, 302)
        send_email.assert_called_once()
        sent_email, reset_url = sent_messages[0]
        self.assertEqual(sent_email, 'user@example.com')
        self.assertTrue(reset_url.startswith('https://sentiment.example.com/reset-password/'))
        raw_token = reset_url.rsplit('/', 1)[-1]

        with self.app.app_context():
            reset_record = PasswordResetToken.query.one()
            self.assertNotEqual(reset_record.token_hash, raw_token)
            self.assertEqual(len(reset_record.token_hash), 64)
            self.assertEqual(reset_record.token_type, 'email')

        reset_page = self.client.get(f'/reset-password/{raw_token}')
        self.assertEqual(reset_page.status_code, 200)
        self.assertEqual(reset_page.headers['Referrer-Policy'], 'no-referrer')
        self.assertEqual(reset_page.headers['Cache-Control'], 'no-store')
        reset_csrf = re.search(
            r'name="csrf_token" value="([^"]+)"',
            reset_page.get_data(as_text=True),
        ).group(1)
        changed = self.client.post(f'/reset-password/{raw_token}', data={
            'csrf_token': reset_csrf,
            'new_password': 'Recovered-Password9!',
            'password_confirm': 'Recovered-Password9!',
        })
        self.assertEqual(changed.status_code, 302)
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertEqual(self.client.get(f'/reset-password/{raw_token}').status_code, 302)
        with self.app.app_context():
            account = self.app.extensions['mongo_db']['users'].find_one({
                'email_normalized': 'user@example.com',
            })
            self.assertTrue(check_password_hash(
                account['password_hash'],
                'Recovered-Password9!',
            ))

        self.client.post('/logout', data={'csrf_token': self._csrf_token('/')})
        login_csrf = self._csrf_token('/login')
        login = self.client.post('/login', data={
            'csrf_token': login_csrf,
            'email': 'user@example.com',
            'password': 'Recovered-Password9!',
        })
        self.assertEqual(login.status_code, 302)

    def test_forgot_password_does_not_disclose_unknown_email(self):
        self._register()
        known_token = self._csrf_token('/forgot-password')
        with patch('sentiment_app.auth._send_password_reset_email'):
            known_response = self.client.post('/forgot-password', data={
                'csrf_token': known_token,
                'email': 'user@example.com',
            })
        known_message = self.client.get(known_response.location).get_data(as_text=True)

        unknown_token = self._csrf_token('/forgot-password')
        with patch('sentiment_app.auth._send_password_reset_email') as send_email:
            unknown_response = self.client.post('/forgot-password', data={
                'csrf_token': unknown_token,
                'email': 'unknown@example.com',
            })
        unknown_message = self.client.get(unknown_response.location).get_data(as_text=True)

        self.assertEqual(known_response.status_code, unknown_response.status_code)
        self.assertIn(
            'If an account exists for that email',
            known_message,
        )
        self.assertIn(
            'If an account exists for that email',
            unknown_message,
        )
        send_email.assert_not_called()

    def test_forgot_password_limits_reset_email_requests(self):
        self._register()
        with patch('sentiment_app.auth._send_password_reset_email') as send_email:
            for _ in range(4):
                response = self.client.post('/forgot-password', data={
                    'csrf_token': self._csrf_token('/forgot-password'),
                    'email': 'user@example.com',
                })
                self.assertEqual(response.status_code, 302)

        self.assertEqual(send_email.call_count, 3)
        with self.app.app_context():
            self.assertEqual(PasswordResetToken.query.count(), 3)

    def test_forgot_password_email_failure_does_not_leave_usable_token(self):
        self._register()
        with patch(
            'sentiment_app.auth._send_password_reset_email',
            side_effect=smtplib.SMTPException('mail server unavailable'),
        ):
            with self.assertLogs('sentiment_app.auth', level='ERROR'):
                response = self.client.post('/forgot-password', data={
                    'csrf_token': self._csrf_token('/forgot-password'),
                    'email': 'user@example.com',
                })

        self.assertEqual(response.status_code, 302)
        with self.app.app_context():
            self.assertEqual(PasswordResetToken.query.count(), 0)

    def test_local_console_otp_resets_password_without_exposing_code_in_http(self):
        self._register()
        self.app.config.update(DEBUG=True, PASSWORD_RESET_CONSOLE_OTP=True)
        with self.assertLogs('sentiment_app.auth', level='WARNING') as captured:
            response = self.client.post('/forgot-password', data={
                'csrf_token': self._csrf_token('/forgot-password'),
                'email': 'user@example.com',
            })

        self.assertEqual(response.status_code, 302)
        log_text = '\n'.join(captured.output)
        otp_match = re.search(r'OTP: (\d{6})', log_text)
        self.assertIsNotNone(otp_match)
        otp = otp_match.group(1)
        self.assertNotIn(otp, response.get_data(as_text=True))
        with self.app.app_context():
            reset_record = PasswordResetToken.query.one()
            user = User.query.filter_by(email='user@example.com').one()
            self.assertEqual(reset_record.token_type, 'otp')
            self.assertNotEqual(reset_record.token_hash, otp)
            self.assertEqual(
                reset_record.token_hash,
                hmac.new(
                    self.app.config['SECRET_KEY'].encode(),
                    f'{user.id}:{otp}'.encode(),
                    hashlib.sha256,
                ).hexdigest(),
            )

        page = self.client.get('/reset-password-otp')
        self.assertEqual(page.status_code, 200)
        self.assertEqual(page.headers['Cache-Control'], 'no-store')
        self.assertIn('server terminal', page.get_data(as_text=True))
        csrf_token = re.search(
            r'name="csrf_token" value="([^"]+)"',
            page.get_data(as_text=True),
        ).group(1)
        reset = self.client.post('/reset-password-otp', data={
            'csrf_token': csrf_token,
            'email': 'user@example.com',
            'otp': otp,
            'new_password': 'Recovered-Password9!',
            'password_confirm': 'Recovered-Password9!',
        })
        self.assertEqual(reset.status_code, 302)
        with self.app.app_context():
            account = self.app.extensions['mongo_db']['users'].find_one({
                'email_normalized': 'user@example.com',
            })
            self.assertTrue(check_password_hash(
                account['password_hash'],
                'Recovered-Password9!',
            ))
            self.assertEqual(account['auth_version'], 1)
            self.assertEqual(PasswordResetToken.query.count(), 0)

    def test_local_console_otp_locks_after_five_invalid_attempts(self):
        self._register()
        self.app.config.update(DEBUG=True, PASSWORD_RESET_CONSOLE_OTP=True)
        with self.assertLogs('sentiment_app.auth', level='WARNING') as captured:
            self.client.post('/forgot-password', data={
                'csrf_token': self._csrf_token('/forgot-password'),
                'email': 'user@example.com',
            })
        otp = re.search(r'OTP: (\d{6})', '\n'.join(captured.output)).group(1)
        csrf_token = self._csrf_token('/reset-password-otp')

        for _ in range(5):
            response = self.client.post('/reset-password-otp', data={
                'csrf_token': csrf_token,
                'email': 'user@example.com',
                'otp': '000000' if otp != '000000' else '000001',
                'new_password': 'Recovered-Password9!',
                'password_confirm': 'Recovered-Password9!',
            })
            self.assertEqual(response.status_code, 200)

        with self.app.app_context():
            reset_record = PasswordResetToken.query.one()
            self.assertEqual(reset_record.failed_attempts, 5)

        rejected = self.client.post('/reset-password-otp', data={
            'csrf_token': csrf_token,
            'email': 'user@example.com',
            'otp': otp,
            'new_password': 'Recovered-Password9!',
            'password_confirm': 'Recovered-Password9!',
        })
        self.assertEqual(rejected.status_code, 200)
        with self.app.app_context():
            account = self.app.extensions['mongo_db']['users'].find_one({
                'email_normalized': 'user@example.com',
            })
            self.assertTrue(check_password_hash(account['password_hash'], self.PASSWORD))

    def test_otp_reset_page_is_unavailable_when_local_mode_is_disabled(self):
        self.assertEqual(self.client.get('/reset-password-otp').status_code, 404)

    def test_local_console_otp_expires(self):
        self._register()
        self.app.config.update(DEBUG=True, PASSWORD_RESET_CONSOLE_OTP=True)
        with self.assertLogs('sentiment_app.auth', level='WARNING') as captured:
            self.client.post('/forgot-password', data={
                'csrf_token': self._csrf_token('/forgot-password'),
                'email': 'user@example.com',
            })
        otp = re.search(r'OTP: (\d{6})', '\n'.join(captured.output)).group(1)
        with self.app.app_context():
            reset_record = PasswordResetToken.query.one()
            reset_record.expires_at = datetime.utcnow() - timedelta(seconds=1)
            db.session.commit()

        response = self.client.post('/reset-password-otp', data={
            'csrf_token': self._csrf_token('/reset-password-otp'),
            'email': 'user@example.com',
            'otp': otp,
            'new_password': 'Recovered-Password9!',
            'password_confirm': 'Recovered-Password9!',
        })
        self.assertEqual(response.status_code, 200)
        with self.app.app_context():
            account = self.app.extensions['mongo_db']['users'].find_one({
                'email_normalized': 'user@example.com',
            })
            self.assertTrue(check_password_hash(account['password_hash'], self.PASSWORD))

    def test_expired_reset_token_is_rejected(self):
        self._register()
        with self.app.app_context():
            user = User.query.filter_by(email='user@example.com').one()
            raw_token = 'expired-test-token'
            db.session.add(PasswordResetToken(
                user_id=user.id,
                token_hash=hashlib.sha256(raw_token.encode()).hexdigest(),
                expires_at=datetime.utcnow() - timedelta(minutes=1),
            ))
            db.session.commit()

        response = self.client.get(f'/reset-password/{raw_token}')
        self.assertEqual(response.status_code, 302)
        with self.app.app_context():
            self.assertEqual(PasswordResetToken.query.count(), 0)

    def test_uploaded_original_is_persisted_and_downloadable(self):
        self._register()
        original = b"An excellent experience.\n"
        analyzed = {
            'input_text': 'An excellent experience.',
            'vader': {
                'sentiment': 'positive',
                'compound': 0.8,
                'pos': 0.8,
                'neg': 0.0,
                'neu': 0.2,
            },
            'textblob': {
                'sentiment': 'positive',
                'polarity': 0.8,
                'subjectivity': 0.5,
            },
            'final_sentiment': 'positive',
            'word_count': 3,
            'sentence_count': 1,
            'top_keywords': ['excellent'],
        }
        with patch('sentiment_app.routes.analyze_text', return_value=analyzed):
            response = self.client.post('/upload', data={
                'csrf_token': self._csrf_token('/upload'),
                'file': (io.BytesIO(original), 'reviews.txt'),
            }, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 302)

        with self.app.app_context():
            uploaded_file = UploadedFile.query.one()
            file_id = uploaded_file.id
            self.assertEqual(uploaded_file.content, original)
            self.assertEqual(uploaded_file.size_bytes, len(original))

        downloaded = self.client.get(f'/file/{file_id}')
        self.assertEqual(downloaded.status_code, 200)
        self.assertEqual(downloaded.data, original)

    def test_uploaded_file_is_not_accessible_to_another_account(self):
        self._register('first@example.com')
        analyzed = {
            'input_text': 'Good.',
            'vader': {
                'sentiment': 'positive',
                'compound': 0.5,
                'pos': 0.5,
                'neg': 0.0,
                'neu': 0.5,
            },
            'textblob': {
                'sentiment': 'positive',
                'polarity': 0.5,
                'subjectivity': 0.5,
            },
            'final_sentiment': 'positive',
            'word_count': 1,
            'sentence_count': 1,
            'top_keywords': ['good'],
        }
        with patch('sentiment_app.routes.analyze_text', return_value=analyzed):
            self.client.post('/upload', data={
                'csrf_token': self._csrf_token('/upload'),
                'file': (io.BytesIO(b'Good.\n'), 'review.txt'),
            }, content_type='multipart/form-data')

        with self.app.app_context():
            uploaded_file = UploadedFile.query.one()
            file_id = uploaded_file.id
            batch_id = uploaded_file.batch_id

        self.client.post('/logout', data={'csrf_token': self._csrf_token('/upload')})
        self._register('second@example.com')
        self.assertEqual(self.client.get(f'/file/{file_id}').status_code, 404)
        self.assertEqual(self.client.get(f'/batch/{batch_id}').status_code, 404)
        self.assertIn('0 Records', self.client.get('/history').get_data(as_text=True))
        self.assertEqual(self.client.get('/api/dashboard').json, {'empty': True})
        self.assertEqual(self.client.get('/export/json').json, [])


class LegacyDatabaseMigrationTests(unittest.TestCase):
    def test_existing_analysis_tables_gain_account_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = os.path.join(temp_dir, 'legacy.db')
            connection = sqlite3.connect(database_path)
            connection.executescript("""
                CREATE TABLE users (
                    id INTEGER PRIMARY KEY,
                    email VARCHAR(254) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NOT NULL,
                    created_at DATETIME NOT NULL
                );
                CREATE TABLE batch_analyses (
                    id INTEGER PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL,
                    file_type VARCHAR(20) NOT NULL,
                    total_records INTEGER,
                    positive_count INTEGER,
                    negative_count INTEGER,
                    neutral_count INTEGER,
                    created_at DATETIME
                );
                CREATE TABLE analysis_results (
                    id INTEGER PRIMARY KEY,
                    input_text TEXT NOT NULL,
                    source_type VARCHAR(50),
                    source_filename VARCHAR(255),
                    vader_sentiment VARCHAR(20),
                    textblob_sentiment VARCHAR(20),
                    final_sentiment VARCHAR(20),
                    vader_compound FLOAT,
                    vader_pos FLOAT,
                    vader_neg FLOAT,
                    vader_neu FLOAT,
                    textblob_polarity FLOAT,
                    textblob_subjectivity FLOAT,
                    word_count INTEGER,
                    sentence_count INTEGER,
                    top_keywords TEXT,
                    batch_id INTEGER,
                    created_at DATETIME
                );
                CREATE TABLE password_reset_tokens (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    token_hash VARCHAR(64) NOT NULL,
                    expires_at DATETIME NOT NULL,
                    created_at DATETIME NOT NULL
                );
                INSERT INTO batch_analyses (id, filename, file_type)
                VALUES (1, 'old.txt', 'txt');
                INSERT INTO analysis_results (id, input_text, batch_id)
                VALUES (1, 'Legacy analysis', 1);
                INSERT INTO users (id, email, password_hash, created_at)
                VALUES (1, 'legacy@example.com', 'legacy-hash', CURRENT_TIMESTAMP);
            """)
            connection.commit()
            connection.close()

            previous_database_url = os.environ.get('DATABASE_URL')
            previous_secret_key = os.environ.get('SECRET_KEY')
            previous_mongodb_database = os.environ.get('MONGODB_DATABASE')
            os.environ['DATABASE_URL'] = f'sqlite:///{database_path}'
            os.environ['SECRET_KEY'] = 'migration-test-secret'
            legacy_mongodb_database = f"sentimentiq_legacy_test_{uuid.uuid4().hex}"
            os.environ['MONGODB_DATABASE'] = legacy_mongodb_database
            app = create_app()
            try:
                with app.app_context():
                    analysis_columns = {
                        column['name']
                        for column in inspect(db.engine).get_columns('analysis_results')
                    }
                    batch_columns = {
                        column['name']
                        for column in inspect(db.engine).get_columns('batch_analyses')
                    }
                    user_columns = {
                        column['name']
                        for column in inspect(db.engine).get_columns('users')
                    }
                    reset_token_columns = {
                        column['name']
                        for column in inspect(db.engine).get_columns('password_reset_tokens')
                    }
                    self.assertIn('user_id', analysis_columns)
                    self.assertIn('groq_sentiment', analysis_columns)
                    self.assertIn('user_id', batch_columns)
                    self.assertIn('auth_version', user_columns)
                    self.assertIn('username', user_columns)
                    self.assertIn('failed_attempts', reset_token_columns)
                    self.assertIn('token_type', reset_token_columns)
                    self.assertEqual(db.session.get(User, 1).auth_version, 0)
                    self.assertEqual(db.session.get(User, 1).username, 'legacy')
                    self.assertIsNone(db.session.get(AnalysisResult, 1).user_id)
                    migrated_account = app.extensions['mongo_db']['users'].find_one({'_id': 1})
                    self.assertEqual(migrated_account['password_hash'], 'legacy-hash')
                    self.assertEqual(
                        db.session.get(User, 1).password_hash,
                        '!managed-in-mongodb!',
                    )
                    db.session.remove()
                    db.engine.dispose()
            finally:
                app.extensions['mongo_client'].drop_database(legacy_mongodb_database)
                app.extensions['mongo_client'].close()
                if previous_database_url is None:
                    os.environ.pop('DATABASE_URL', None)
                else:
                    os.environ['DATABASE_URL'] = previous_database_url
                if previous_secret_key is None:
                    os.environ.pop('SECRET_KEY', None)
                else:
                    os.environ['SECRET_KEY'] = previous_secret_key
                if previous_mongodb_database is None:
                    os.environ.pop('MONGODB_DATABASE', None)
                else:
                    os.environ['MONGODB_DATABASE'] = previous_mongodb_database


if __name__ == '__main__':
    unittest.main()
