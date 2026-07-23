# SPDX-License-Identifier: Apache-2.0
# Part of rsymbolic2, Copyright 2026 Toshihiro Iguchi.
#
# Local development server for web/app.
#
# `python -m http.server` sends no Cache-Control, so browsers fall back to heuristic
# freshness (a fraction of the file's age) and happily serve a stale index.html or, worse,
# a stale vendor/rsymbolic2.wasm for hours after a rebuild — the page then runs the OLD
# engine with the NEW UI and the mismatch is silent. This server sends no-store on
# everything so every reload picks up the current build.
#
# Development only. Deployment is GitHub Pages (see README), which sets its own headers.
#
#   python web/serve.py [port]     # default 8080, serves web/app

import http.server
import os
import sys

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=APP_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), NoCacheHandler) as httpd:
        print(f"serving {APP_DIR} (no-store) at http://localhost:{port}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
