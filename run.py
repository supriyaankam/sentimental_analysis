"""Entry point — run this file to start the development server."""

import os
from pathlib import Path
import sys
import urllib.error
import urllib.request
import webbrowser


PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / '.venv'
VENV_PYTHON = VENV_DIR / 'Scripts' / 'python.exe'

# If run from a system Python prompt, use the project's pinned environment.
if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != VENV_DIR.resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])


SERVER_URL = 'http://127.0.0.1:5000/'
APP_SHELL_MARKER = 'data-app-shell="sentimentiq-username-nav-v1"'


def _existing_server_is_this_app():
    try:
        with urllib.request.urlopen(SERVER_URL, timeout=2) as response:
            page = response.read(65536).decode('utf-8', errors='replace')
            return response.status == 200 and APP_SHELL_MARKER in page
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


if __name__ == '__main__' and _existing_server_is_this_app():
    print(f'SentimentIQ is already running at {SERVER_URL}')
    webbrowser.open(SERVER_URL)
    raise SystemExit(0)


from sentiment_app import create_app

app = create_app()


if __name__ == '__main__':
    try:
        app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
    except OSError as error:
        if _existing_server_is_this_app():
            print(f'SentimentIQ is already running at {SERVER_URL}')
            webbrowser.open(SERVER_URL)
            raise SystemExit(0)
        raise RuntimeError(
            'Could not start SentimentIQ on port 5000. '
            'Close the program using that port, then run python run.py again.'
        ) from error
