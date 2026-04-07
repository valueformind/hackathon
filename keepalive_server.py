"""
Minimal keepalive server for Hugging Face Spaces.

Why this exists:
- `python -m http.server` returns 501 for POST requests.
- Some HF infra probes may send POST (e.g., `/reset`).
- This server accepts GET/HEAD/POST and always returns 200 JSON.
"""

import json
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

    def log_message(self, fmt: str, *args) -> None:
        # Keep logs clean; inference stdout format must remain strict.
        return


def main() -> None:
    server = HTTPServer(("0.0.0.0", 7860), KeepAliveHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
