#!/usr/bin/env python3
"""
extract_meteo.py — Open-Meteo 14-day cloud + visibility (fog) forecast grid for the
Iceland Sky Predictor.

Runs on GitHub Actions (server-side): fetches the full Iceland DEFAULT box at 0.20°
(~22 km) resolution — 36×64 = 2304 points — with 12-second pauses between 150-point
chunks so the per-minute rate limit (600 locations/min) is never hit. Outputs a single
JSON that the app loads from GitHub Pages (zero Open-Meteo calls from the user's device).

The output mirrors the structure that clouds.js `makeSampler()` expects: an array of
point objects with {lat, lon, time:[], low:[], mid:[], high:[], visibility:[]}.

Usage:  python extract_meteo.py [--out FILE] [--step DEGREES] [--days N]
        Defaults: --out meteo_latest.json  --step 0.20  --days 14
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("pip install requests")

API = "https://api.open-meteo.com/v1/forecast"
HOURLY = "cloud_cover_low,cloud_cover_mid,cloud_cover_high,visibility"
CHUNK = 150          # max locations per HTTP request
DELAY = 20           # seconds between chunks (stay well under 600 locs/min)
BOX = dict(lat0=63.0, lat1=69.9, lon0=-25.2, lon1=-12.6)


def grid_coords(step: float):
    """Generate the regular lat/lon grid (matches clouds.js gridCoords)."""
    def axis(a, b, s):
        n = max(2, round((b - a) / s) + 1)
        return [a + (b - a) * i / (n - 1) for i in range(n)]
    lats_ax = axis(BOX["lat0"], BOX["lat1"], step)
    lons_ax = axis(BOX["lon0"], BOX["lon1"], step)
    lats, lons = [], []
    for la in lats_ax:
        for lo in lons_ax:
            lats.append(round(la, 4))
            lons.append(round(lo, 4))
    return lats, lons


def fetch_chunk(lats: list, lons: list, start: str, end: str, retries: int = 2) -> list:
    """Fetch one chunk of ≤150 locations from Open-Meteo. Returns list of point dicts."""
    params = {
        "latitude": ",".join(f"{x:.4f}" for x in lats),
        "longitude": ",".join(f"{x:.4f}" for x in lons),
        "hourly": HOURLY,
        "timezone": "UTC",
        "start_date": start,
        "end_date": end,
    }
    for attempt in range(retries + 1):
        resp = requests.get(API, params=params, timeout=60)
        if resp.status_code == 429:
            if attempt < retries:
                wait = 65  # wait >1 minute for the per-minute window to reset
                print(f" 429 (rate limit), waiting {wait}s...", end="", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError("Open-Meteo 429 rate limit hit after retries — try later.")
        resp.raise_for_status()
        break
    data = resp.json()
    if not isinstance(data, list):
        data = [data]
    points = []
    for it in data:
        h = it.get("hourly", {})
        points.append({
            "lat": it["latitude"],
            "lon": it["longitude"],
            "time": h.get("time", []),
            "low": h.get("cloud_cover_low", []),
            "mid": h.get("cloud_cover_mid", []),
            "high": h.get("cloud_cover_high", []),
            "visibility": h.get("visibility", []),
        })
    return points


def main():
    ap = argparse.ArgumentParser(description="Fetch Open-Meteo cloud+fog grid for Iceland")
    ap.add_argument("--out", default="meteo_latest.json", help="Output file path")
    ap.add_argument("--step", type=float, default=0.20, help="Grid step in degrees (default 0.20)")
    ap.add_argument("--days", type=int, default=14, help="Forecast days (default 14)")
    args = ap.parse_args()

    lats, lons = grid_coords(args.step)
    n_pts = len(lats)
    n_chunks = -(-n_pts // CHUNK)  # ceiling division
    print(f"Grid: step={args.step} deg -> {n_pts} points in {n_chunks} chunks")

    today = dt.date.today().isoformat()
    end = (dt.date.today() + dt.timedelta(days=args.days - 1)).isoformat()
    print(f"Time range: {today} -> {end} ({args.days} days)")

    all_points = []
    for i in range(0, n_pts, CHUNK):
        chunk_lats = lats[i:i + CHUNK]
        chunk_lons = lons[i:i + CHUNK]
        chunk_idx = i // CHUNK + 1
        print(f"  chunk {chunk_idx}/{n_chunks} ({len(chunk_lats)} pts)...", end="", flush=True)
        pts = fetch_chunk(chunk_lats, chunk_lons, today, end)
        all_points.extend(pts)
        print(f" ok ({len(pts[0]['time'])} hours)")
        if i + CHUNK < n_pts:
            time.sleep(DELAY)

    # Build output JSON (compact: arrays of values, shared time axis)
    # All points share the same time array (same start/end), so store it once at top level.
    times = all_points[0]["time"] if all_points else []
    output = {
        "source": "Open-Meteo forecast (server-side, pre-built)",
        "generated": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"),
        "step": args.step,
        "days": args.days,
        "start_date": today,
        "end_date": end,
        "lat0": BOX["lat0"], "lat1": BOX["lat1"],
        "lon0": BOX["lon0"], "lon1": BOX["lon1"],
        "times": times,
        "points": [{
            "lat": p["lat"], "lon": p["lon"],
            "low": p["low"], "mid": p["mid"], "high": p["high"],
            "visibility": p["visibility"],
        } for p in all_points],
    }

    # Atomic write: write to temp, rename on success
    out_path = Path(args.out)
    tmp_path = out_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output, f, separators=(",", ":"))
    tmp_path.rename(out_path)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nDone: {out_path} ({size_mb:.1f} MB, {len(all_points)} points, {len(times)} hours)")


if __name__ == "__main__":
    main()
