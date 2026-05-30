"""
SEAL PPG webapp entry point.

Single-command launcher for the acquisition + analysis dashboard:

    python app.py

Brings up the FastAPI server on http://127.0.0.1:8000 and opens the
default browser to the dashboard. Recording subprocesses, signal
loading, SQI analysis and the per-session Bland-Altman plots are all
served from the same process — no separate scripts to start.

Optional environment variables:
    SEAL_WEBAPP_HOST   default 127.0.0.1
    SEAL_WEBAPP_PORT   default 8000
    SEAL_WEBAPP_NOOPEN if set, skip auto-opening the browser
"""

import os

# Force matplotlib's headless Agg backend before any webapp module (which
# imports pyhrv/pingouin, and through them pyplot) loads. Analysis runs on
# uvicorn worker threads; the default Tk backend would create Tk objects off
# the main thread and crash with Tcl_AsyncDelete. See webapp/analysis.py.
os.environ.setdefault("MPLBACKEND", "Agg")

import threading
import webbrowser

import uvicorn

from webapp.api import app   # re-exported here for `uvicorn app:app` users too


def _open_browser(url, delay_s=1.2):
    def _go():
        try:
            webbrowser.open(url)
        except Exception:
            pass
    threading.Timer(delay_s, _go).start()


def main():
    host = os.environ.get("SEAL_WEBAPP_HOST", "127.0.0.1")
    port = int(os.environ.get("SEAL_WEBAPP_PORT", "8000"))
    url = f"http://{host}:{port}/"

    print("=" * 62)
    print("  SEAL PPG -- Acquisition & Analysis Dashboard")
    print("=" * 62)
    print(f"  Open in browser: {url}")
    print(f"  Receiver script: PPG_ECG_Full_Unpacking.py")
    print(f"  Sessions dir   : {os.path.dirname(os.path.abspath(__file__))}")
    print("  Ctrl+C to stop the server.")
    print("=" * 62)

    if not os.environ.get("SEAL_WEBAPP_NOOPEN"):
        _open_browser(url)

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
