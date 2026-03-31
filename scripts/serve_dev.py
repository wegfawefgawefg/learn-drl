#!/usr/bin/env python3

import argparse
import html
import mimetypes
import os
import posixpath
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_PATH = "/learn-drl"
WATCH_INTERVAL = 0.75

HOT_RELOAD_SNIPPET = f"""
<script>
(() => {{
  const versionUrl = "{BASE_PATH}/__dev_version";
  let currentVersion = null;

  async function checkVersion() {{
    try {{
      const response = await fetch(versionUrl, {{ cache: "no-store" }});
      if (!response.ok) return;

      const nextVersion = await response.text();
      if (currentVersion === null) {{
        currentVersion = nextVersion;
        return;
      }}

      if (nextVersion !== currentVersion) {{
        window.location.reload();
      }}
    }} catch (_error) {{
      // Ignore brief disconnects during restarts.
    }}
  }}

  checkVersion();
  window.setInterval(checkVersion, 1000);
}})();
</script>
"""


class ChangeTracker:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.lock = threading.Lock()
        self.version = 0
        self.snapshot = self._take_snapshot()

    def _take_snapshot(self) -> dict[str, tuple[int, int]]:
        snapshot: dict[str, tuple[int, int]] = {}
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            if ".git" in path.parts:
                continue

            try:
                stat = path.stat()
            except OSError:
                continue

            snapshot[str(path.relative_to(self.root))] = (stat.st_mtime_ns, stat.st_size)
        return snapshot

    def maybe_bump(self) -> None:
        next_snapshot = self._take_snapshot()
        with self.lock:
            if next_snapshot != self.snapshot:
                self.snapshot = next_snapshot
                self.version += 1

    def get_version(self) -> int:
        with self.lock:
            return self.version


class SiteHandler(BaseHTTPRequestHandler):
    server_version = "LearnDRLDevServer/1.0"

    def do_GET(self) -> None:
        self._handle_request(send_body=True)

    def do_HEAD(self) -> None:
        self._handle_request(send_body=False)

    def _handle_request(self, send_body: bool) -> None:
        request_path = urlsplit(self.path).path

        if request_path == "/":
            self._redirect(f"{BASE_PATH}/")
            return
        if request_path == BASE_PATH:
            self._redirect(f"{BASE_PATH}/")
            return
        if request_path == f"{BASE_PATH}/__dev_version":
            self._serve_version()
            return
        if not request_path.startswith(f"{BASE_PATH}/"):
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        relative_path = request_path[len(BASE_PATH) :]
        file_path = self._resolve_path(relative_path)
        if file_path is None:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        if file_path.is_dir():
            file_path = file_path / "index.html"
        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        if file_path.suffix.lower() == ".html":
            self._serve_html(file_path, send_body)
            return
        self._serve_file(file_path, send_body)

    def log_message(self, fmt: str, *args) -> None:
        print(
            "%s - - [%s] %s"
            % (self.address_string(), self.log_date_time_string(), fmt % args)
        )

    def _redirect(self, location: str) -> None:
        self.send_response(HTTPStatus.MOVED_PERMANENTLY)
        self.send_header("Location", location)
        self.end_headers()

    def _serve_version(self) -> None:
        version = str(self.server.change_tracker.get_version()).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(version)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(version)

    def _resolve_path(self, relative_path: str) -> Path | None:
        normalized = posixpath.normpath(unquote(relative_path))
        if normalized in ("", "."):
            normalized = "/"
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"

        candidate = (REPO_ROOT / normalized.lstrip("/")).resolve()
        if REPO_ROOT not in candidate.parents and candidate != REPO_ROOT:
            return None
        return candidate

    def _serve_html(self, file_path: Path, send_body: bool) -> None:
        raw_html = file_path.read_text(encoding="utf-8")
        if "</body>" in raw_html:
            body = raw_html.replace("</body>", f"{HOT_RELOAD_SNIPPET}\n</body>", 1)
        else:
            body = f"{raw_html}\n{HOT_RELOAD_SNIPPET}\n"

        payload = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if send_body:
            self.wfile.write(payload)

    def _serve_file(self, file_path: Path, send_body: bool) -> None:
        payload = file_path.read_bytes()
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if send_body:
            self.wfile.write(payload)


def watch_for_changes(change_tracker: ChangeTracker) -> None:
    while True:
        change_tracker.maybe_bump()
        time.sleep(WATCH_INTERVAL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Serve the site locally at http://localhost:8000/learn-drl/ with "
            "automatic browser reloads when files change."
        )
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind. Default: 127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind. Default: 8000",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    change_tracker = ChangeTracker(REPO_ROOT)

    watcher = threading.Thread(
        target=watch_for_changes,
        args=(change_tracker,),
        daemon=True,
    )
    watcher.start()

    server = ThreadingHTTPServer((args.host, args.port), SiteHandler)
    server.change_tracker = change_tracker  # type: ignore[attr-defined]

    url = f"http://{args.host}:{args.port}{BASE_PATH}/"
    print(f"Serving {html.escape(str(REPO_ROOT))} at {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
