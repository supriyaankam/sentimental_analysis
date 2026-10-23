# SentimentIQ — NLP Sentiment Analysis System

A full-stack web application for sentiment analysis powered by NLTK, VADER, TextBlob, and optional Groq analysis.

---

## Features

| Feature | Details |
|---|---|
| Text Analysis | Type or paste any text for instant sentiment analysis |
| File Upload | Upload TXT, PDF, DOCX, CSV, JSON, XLSX/XLS |
| Sentiment Analysis | VADER and TextBlob, with optional Groq AI analysis for single text |
| Visual Dashboard | Pie chart, trend line, bar comparison, radar, word frequency |
| Word Cloud | Generated per analysis and per batch |
| POS Tagging | Part-of-speech tags shown on result page |
| History | Paginated, filterable, searchable result history |
| Export | Download all history as CSV or JSON |
| REST API | Authenticated POST `/api/analyze` for programmatic access |
| Accounts | Register and log in with an email address and password |
| Database | MongoDB stores account credentials; SQLite via SQLAlchemy stores analysis results and uploaded originals |
| Privacy | Analysis history, uploaded files, reports, and dashboard data are scoped to the signed-in account |

---

## Quick Start

```bash
# 1. Install dependencies (already done if you ran setup)
python -m pip install -r requirements.txt

# 2. Ensure the local MongoDB Community Server service is running.
#    Compass can connect to the same server at mongodb://localhost:27017.

# 3. Create .env from .env.example and set SECRET_KEY to a random value
#    Generate one with: python -c "import secrets; print(secrets.token_hex(32))"

# 4. Start the server
python run.py
```

The app connects to `mongodb://localhost:27017` by default and creates the
`sentimentiq` database with a `users` collection. If MongoDB uses authentication
or a different host, set `MONGODB_URI` in `.env`. The Flask process reuses one
PyMongo client and uses the driver's default connection pool settings.

Create an account in the browser with a username and email address. The username
is shown in the navigation bar after signing in. Passwords must be 12-128
characters and include uppercase and lowercase letters, a number, and a special
character. Use the Logout button in the navigation bar to end a session.
Set a stable random
`SECRET_KEY` in `.env`; changing it invalidates existing login sessions.
Passwords are stored as salted hashes in MongoDB, never as plain text. In
Compass, connect to `mongodb://localhost:27017`, open `sentimentiq`, then
`users`. Account documents contain `_id`, `email`, `email_normalized`,
`password_hash`, `auth_version`, timestamps, and `schema_version`. SQLite keeps
the account ID/email link used to associate analysis rows; the old SQLite
password hashes are migrated to MongoDB on startup and replaced with unusable
sentinel values. Analysis results and uploaded source file blobs remain in SQLite.
Deleting only a document from MongoDB Compass does not delete its SQLite account
identity. Registering that email again restores credentials against the existing
identity and retains its analysis history. Startup migration does not recreate a
deleted MongoDB account from the SQLite sentinel value.

### Forgot password

The login page includes a password recovery link. For real deployments, configure
`MAIL_SERVER`, `MAIL_PORT`, `MAIL_USE_TLS` or `MAIL_USE_SSL`, `MAIL_USERNAME`,
`MAIL_PASSWORD`, and `MAIL_DEFAULT_SENDER` in `.env`. Set `PUBLIC_BASE_URL` to
the app's canonical public origin (for example, `https://sentiment.example.com`)
so reset links are generated for the correct site. Reset links expire after 30
minutes, can only be used once, and the database stores only their token hashes.
Reset requests are limited to three per account every 15 minutes. The recovery
page gives the same response whether or not the email is registered.

For local development without working email, set
`PASSWORD_RESET_CONSOLE_OTP=true` in `.env` and run the app with `python run.py`.
When Flask debug mode is on, the recovery request prints a six-digit OTP only to
the server terminal; enter it at the OTP reset page linked from the recovery
screen. The code expires after 30 minutes, is stored as a keyed hash, is
single-use, and allows five attempts. This local-only flow requires Flask debug
mode as well as the explicit setting; it is disabled by default and must never
be enabled in production. A production deployment should use email or another
verified delivery channel because a terminal OTP is not a remote account
recovery mechanism.

### Optional Groq analysis

Groq adds an AI sentiment label and explanation to single-text and authenticated
API analyses. Set `GROQ_API_KEY` in the project-root `.env` file (and optionally
`GROQ_MODEL`), then restart the app so Flask loads the setting. The key stays
server-side and `.env` is ignored by Git. Requests use JSON response mode and a
bounded timeout; failures are recorded in the server log and shown on the
analysis result without exposing credentials. Batch file analysis continues to
use the local sentiment engines. Without a key, local sentiment analysis
continues to work.

Then open your browser at: **http://127.0.0.1:5000**

## Deploy to Render

This repository includes a `render.yaml` Blueprint for deploying the Flask app
to Render with a MongoDB Atlas account database and a persistent disk for the
SQLite analysis database. The web service uses Render's paid Starter plan
because persistent disks are not available on the Free plan. The single Gunicorn
worker is intentional: SQLite should not be shared by multiple service instances.

1. Push the project to a GitHub repository, then create a MongoDB Atlas cluster
   and database user. Allow the Render service to reach Atlas in Network Access;
   restrict allowed IPs where your Render plan supports stable outbound IPs.
2. In Render, choose **New > Blueprint**, connect the repository, and deploy
   its `render.yaml` file.
3. When prompted for `MONGODB_URI`, enter the Atlas connection string for the
   `sentimentiq` database. Keep the database user's credentials in that URI
   encoded as required by MongoDB; never commit the URI to the repository.
4. Wait for the build and health check to pass, then open the `onrender.com`
   URL shown in the Render dashboard and create an account.

The Blueprint creates a random `SECRET_KEY`, installs the NLTK resources used
by the analysis pipeline, enables secure session cookies, and stores SQLite
data under the persistent disk at `/var/data`. Do not remove that disk or change
`DATABASE_URL` to a path outside it, or stored analysis data will not persist.
Back up the disk and MongoDB data before making significant changes.

Password-reset email is not configured by default. To enable it, set
`MAIL_SERVER`, `MAIL_PORT`, `MAIL_USE_TLS` or `MAIL_USE_SSL`,
`MAIL_USERNAME`, `MAIL_PASSWORD`, `MAIL_DEFAULT_SENDER`, and `PUBLIC_BASE_URL`
in the Render service's Environment settings. `PUBLIC_BASE_URL` should be the
HTTPS origin of the deployed app. `GROQ_API_KEY` is also optional and can be
set there to enable AI explanations.

---

## Project Structure

```
nlp_project/
├── run.py                        ← Entry point
├── requirements.txt
├── .env.example                  ← MongoDB and email connection settings
├── sentiment.db                  ← SQLite database (analysis results and uploaded files)
└── sentiment_app/
    ├── __init__.py               ← Flask app factory
    ├── extensions.py             ← SQLAlchemy, Migrate
    ├── routes.py                 ← All URL routes
    ├── auth.py                   ← Account registration, login, and logout
    ├── nlp_engine.py             ← VADER + TextBlob
    ├── document_parser.py        ← TXT/PDF/DOCX/CSV/JSON/Excel parsers
    ├── chart_utils.py            ← Word cloud + chart data helpers
    ├── models/
    │   └── database.py           ← User, AnalysisResult, BatchAnalysis, UploadedFile models
    ├── templates/
    │   ├── auth/                 ← Login, registration, and password recovery pages
    │   ├── base.html             ← Navbar, flash messages, footer
    │   ├── index.html            ← Home + text input
    │   ├── result.html           ← Single analysis result
    │   ├── upload.html           ← File upload page
    │   ├── batch_result.html     ← Batch/file analysis result
    │   ├── history.html          ← Paginated history
    │   └── dashboard.html        ← Visual analytics dashboard
    ├── static/
    │   └── css/style.css         ← Full dark-theme stylesheet
    └── uploads/                  ← Temporary processing files and generated exports
```

---

## REST API

**Analyse a single text:**
```bash
curl -X POST http://127.0.0.1:5000/api/analyze \
     -H "Content-Type: application/json" \
     -H "X-CSRF-Token: <token-from-login-session>" \
     -d "{\"text\": \"I absolutely love this product!\"}"
```

The API requires an authenticated session. Log in through the web interface,
retain the session cookie, and send the CSRF token from the page as the
`X-CSRF-Token` header on POST requests. API results are saved to that account.

**Get dashboard data (JSON):**
```
GET http://127.0.0.1:5000/api/dashboard
```

Existing analysis rows from before account support are left unassigned and are
not shown to new accounts.

---

## Supported File Formats

| Format | How text is extracted |
|---|---|
| `.txt` | Split by paragraphs / lines |
| `.pdf` | Per-page text extraction (PyPDF2) |
| `.docx` | Paragraphs + table cells |
| `.csv` | Auto-detects `text/review/comment` column |
| `.json` | Recursively extracts string values |
| `.xlsx/.xls` | All sheets, smart column detection |
