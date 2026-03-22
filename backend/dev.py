"""
Dev server runner with reliable auto-restart on file changes.

Usage:  python dev.py

Watches all .py files in the backend directory (excluding dev.py itself,
tests/, and venv/). Restarts the server when any watched file changes.
"""

import subprocess
import sys
import os
import time
import hashlib
import socket
from pathlib import Path

BACKEND_DIR = Path(__file__).parent
PORT = 8000
COOLDOWN_SECONDS = 5    # ignore changes for this long after a (re)start
PORT_FREE_TIMEOUT = 8   # seconds to wait for port to be released after kill

# Files/dirs to ignore when watching for changes
SKIP = {"dev.py", "venv", ".venv", "__pycache__", "tests", ".git"}


def hash_files(directory: Path) -> str:
    """Compute a combined MD5 of all watched .py files."""
    h = hashlib.md5()
    for path in sorted(directory.glob("**/*.py")):
        if any(s in str(path) for s in SKIP):
            continue
        try:
            h.update(path.read_bytes())
            h.update(str(path).encode())
        except Exception:
            pass
    return h.hexdigest()


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_for_port_free(port: int, timeout: int) -> bool:
    """Wait up to `timeout` seconds for the port to be released."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_is_free(port):
            return True
        time.sleep(0.5)
    return False


def start_server() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--port", str(PORT)],
        cwd=BACKEND_DIR,
    )


def main():
    os.chdir(BACKEND_DIR)

    if not port_is_free(PORT):
        print(f"[dev] Port {PORT} is in use by another process.")
        print(f"[dev] Stop the old server first (Ctrl+C in that terminal), then re-run dev.py.")
        sys.exit(1)

    print(f"[dev] Starting server on http://localhost:{PORT}")
    print(f"[dev] Watching {BACKEND_DIR} for .py changes (cooldown: {COOLDOWN_SECONDS}s)...")

    proc = start_server()
    last_hash = hash_files(BACKEND_DIR)
    last_start = time.time()

    try:
        while True:
            time.sleep(1)

            # Restart if server crashed
            if proc.poll() is not None:
                print("[dev] Server stopped unexpectedly — restarting...")
                wait_for_port_free(PORT, PORT_FREE_TIMEOUT)
                proc = start_server()
                last_hash = hash_files(BACKEND_DIR)
                last_start = time.time()
                continue

            # Skip change detection during cooldown window
            if time.time() - last_start < COOLDOWN_SECONDS:
                continue

            current_hash = hash_files(BACKEND_DIR)
            if current_hash != last_hash:
                print("\n[dev] Change detected — restarting server...")
                proc.terminate()
                proc.wait()

                freed = wait_for_port_free(PORT, PORT_FREE_TIMEOUT)
                if not freed:
                    print(f"[dev] Warning: port {PORT} still busy after {PORT_FREE_TIMEOUT}s, trying anyway...")

                proc = start_server()
                last_hash = current_hash
                last_start = time.time()
                print(f"[dev] Server restarted on http://localhost:{PORT}\n")

    except KeyboardInterrupt:
        print("\n[dev] Stopping server...")
        proc.terminate()
        proc.wait()
        print("[dev] Done.")


if __name__ == "__main__":
    main()
