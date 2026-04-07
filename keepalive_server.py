"""
Minimal keepalive server for Hugging Face Spaces.

Why this exists:
- `python -m http.server` returns 501 for POST requests.
- Some HF infra probes may send POST (e.g., `/reset`).
- This server accepts GET/HEAD/POST and always returns 200 JSON.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class KeepAliveHandler(BaseHTTPRequestHandler):
    def _write_ok(self) -> None:
        body = json.dumps({"status": "ok"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        self._write_ok()

    def do_HEAD(self) -> None:  # noqa: N802
        self.send_response(200)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        # Drain request body (if any) to keep the connection healthy.
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length > 0:
            _ = self.rfile.read(length)
        self._write_ok()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("Allow", "GET, HEAD, POST, OPTIONS, PUT, PATCH, DELETE")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS, PUT, PATCH, DELETE"
        )
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Content-Length", "0")
        self.end_headers()

    # Some platform probes may use verbs beyond GET/HEAD/POST.
    # Treat them as liveness checks and return 200.
    def do_PUT(self) -> None:  # noqa: N802
        self.do_POST()

    def do_PATCH(self) -> None:  # noqa: N802
        self.do_POST()

    def do_DELETE(self) -> None:  # noqa: N802
        self._write_ok()

    def log_message(self, fmt: str, *args) -> None:
        # Keep logs clean; inference stdout format must remain strict.
        return


def main() -> None:
    port = int(os.getenv("PORT", "7860"))
    print(
        f"[keepalive] starting file={__file__} pid={os.getpid()} bind=0.0.0.0:{port}",
        file=sys.stderr,
        flush=True,
    )
    server = HTTPServer(("0.0.0.0", port), KeepAliveHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
