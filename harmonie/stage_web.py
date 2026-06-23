#!/usr/bin/env python3
"""Stage the offline PWA web app into _site for GitHub Pages, next to the data JSON.

Layout produced (served at https://<user>.github.io/<repo>/):
  _site/index.html, _site/sw.js                 -> app shell at the Pages root
  _site/static/...                              -> app.js, style.css, js/, vendor/,
                                                   data/, icons/, manifest.webmanifest
  _site/harmonie_latest.json, elements_latest.json (written separately by extract.py)

Done in Python (not inline shell) so a whitespace/indentation hiccup in the workflow
can never turn this into a broken bash script. Never hard-fails the deploy: if the
web/ app isn't present it prints a warning and publishes data-only.
"""
import os
import shutil
import sys

WEB = "web"
SITE = "_site"
SHELL_AT_ROOT = ("index.html", "sw.js")
DROP_FROM_STATIC = ("harmonie_latest.json", "elements_latest.json")


def stage(site: str) -> int:
    static = os.path.join(site, "static")
    if not os.path.isfile(os.path.join(WEB, "index.html")):
        print("::warning::web/index.html not found - publishing data only.")
        print("Repo top-level:", sorted(os.listdir(".")))
        if os.path.isdir(WEB):
            print("web/ contains:", sorted(os.listdir(WEB)))
        return 0

    os.makedirs(static, exist_ok=True)
    for name in sorted(os.listdir(WEB)):
        src = os.path.join(WEB, name)
        dst = os.path.join(static, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    for name in SHELL_AT_ROOT:
        src = os.path.join(static, name)
        if os.path.isfile(src):
            shutil.move(src, os.path.join(site, name))

    for name in DROP_FROM_STATIC:
        p = os.path.join(static, name)
        if os.path.isfile(p):
            os.remove(p)

    print("Staged web app.")
    print(site, "root :", sorted(os.listdir(site)))
    print(site, "static:", sorted(os.listdir(static)))
    return 0


def main(argv) -> int:
    # Default output is _site (GitHub Pages, the workflow path). Pass --out DIR to stage
    # the SAME layout elsewhere — e.g. `--out www` for the Capacitor APK webDir.
    site = SITE
    i = 0
    while i < len(argv):
        if argv[i] == "--out":
            site = argv[i + 1]; i += 2
        else:
            i += 1
    return stage(site)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
