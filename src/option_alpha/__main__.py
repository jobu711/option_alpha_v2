"""Entry point for python -m option_alpha.

Zero-config startup: runs the FastAPI dashboard server on localhost,
initializes the database, and opens the browser automatically.
"""

import logging
import signal
import sys
import threading
import webbrowser

logger = logging.getLogger("option_alpha")

HOST = "127.0.0.1"
PORT = 8000
BANNER = f"""\
============================================
  Option Alpha v1.0 - AI Options Scanner
============================================
  Starting at http://{HOST}:{PORT}
  Press Ctrl+C to stop
============================================
"""


def _open_browser_delayed(url: str, delay: float = 1.5) -> None:
    """Open the browser after a short delay to let the server start."""
    import time

    time.sleep(delay)
    try:
        webbrowser.open(url)
    except Exception:
        pass  # Non-critical; user can open manually


def main() -> None:
    """Run the Option Alpha dashboard server."""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required. Install with: pip install uvicorn[standard]")
        sys.exit(1)

    print(BANNER)

    url = f"http://{HOST}:{PORT}"

    # Open browser in a background thread so it doesn't block server startup.
    browser_thread = threading.Thread(
        target=_open_browser_delayed, args=(url,), daemon=True
    )
    browser_thread.start()

    # Handle graceful shutdown on Ctrl+C.
    def _shutdown_handler(signum, frame):
        print("\nShutting down Option Alpha...")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown_handler)

    # Configure logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Start the uvicorn server (blocking).
    # Security: bind to 127.0.0.1 only (NOT 0.0.0.0).
    uvicorn.run(
        "option_alpha.web.app:create_app",
        host=HOST,
        port=PORT,
        factory=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
