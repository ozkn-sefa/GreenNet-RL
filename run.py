"""
Main Launcher for Green SDN RL Simulation & Interactive Web Dashboard.
Usage:
    python run.py --host 127.0.0.1 --port 8000
"""

import argparse
import webbrowser
import threading
import time
import sys
import uvicorn

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.web.server import app


def open_browser(url: str):
    time.sleep(1.5)
    print(f"\n[*] Web Tarayicisi aciliyor: {url}")
    webbrowser.open(url)


def main():
    parser = argparse.ArgumentParser(description="Start Green SDN RL Simulator Dashboard")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print("=" * 70)
    print(" [GREEN SDN] RL-DESTEKLI AKILLI AG VE ENERJI SIMULATORU")
    print(f" [*] Dashboard URL : {url}")
    print(" [*] Telemetry WS  : ws://127.0.0.1:8000/ws/telemetry")
    print("=" * 70)

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(url,), daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
