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
CHUNK = 75           # points per request (smaller chunks = less server compute = fewer timeouts with 14-day range)
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


SESSION = requests.Session()
# On GitHub runners a request occasionally hangs with no reply while an immediate retry
# answers in ~1 s (a healthy 75-point 14-day chunk takes ~1.5 s), so give up quickly and
# retry rather than sitting out a long read timeout. (connect, read) seconds.
TIMEOUT = (15, 45)


class _Retry(Exception):
    """A reply we should simply ask for again (rate limit, server error, broken body)."""


def fetch_chunk(lats: list, lons: list, start: str, end: str, retries: int = 6) -> list:
    """Fetch one chunk of locations from Open-Meteo. Returns list of point dicts.

    Everything transient is retried: timeouts/stalls, dropped connections, 429, 5xx, and
    replies that arrive as "200 OK" but with an empty or truncated JSON body (seen on GitHub
    runners; this used to crash the whole run). A reply is only accepted when it has exactly
    one entry per requested point with the full hour range for every variable."""
    params = {
        "latitude": ",".join(f"{x:.4f}" for x in lats),
        "longitude": ",".join(f"{x:.4f}" for x in lons),
        "hourly": HOURLY,
        "timezone": "UTC",
        "start_date": start,
        "end_date": end,
    }
    exp_hours = ((dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days + 1) * 24
    keys = HOURLY.split(",")
    last = None
    for attempt in range(retries + 1):
        wait = 5 if attempt < 2 else 30
        try:
            resp = SESSION.get(API, params=params, timeout=TIMEOUT)
            if resp.status_code == 429:
                wait = 65  # wait >1 minute for the per-minute window to reset
                raise _Retry("429 rate limit")
            if resp.status_code >= 500:
                raise _Retry(f"{resp.status_code} server error")
            resp.raise_for_status()          # other 4xx = a bad request: retrying won't help
            data = resp.json()               # ValueError if the body is empty/truncated
            if not isinstance(data, list):
                data = [data]
            if len(data) != len(lats):
                raise _Retry(f"incomplete reply: {len(data)}/{len(lats)} points")
            for it in data:
                h = it.get("hourly") or {}
                if len(h.get("time", [])) != exp_hours or any(len(h.get(k) or []) != exp_hours for k in keys):
                    raise _Retry(f"incomplete reply: {len(h.get('time', []))}/{exp_hours} hours")
        except requests.exceptions.HTTPError:
            raise
        except (requests.exceptions.RequestException, ValueError, _Retry) as e:
            last = e
            if attempt == retries:
                break
            print(f" {type(e).__name__}: {str(e)[:60]}; retry in {wait}s...", end="", flush=True)
            time.sleep(wait)
            continue
        return [{
            "lat": it["latitude"],
            "lon": it["longitude"],
            "time": it["hourly"]["time"],
            "low": it["hourly"]["cloud_cover_low"],
            "mid": it["hourly"]["cloud_cover_mid"],
            "high": it["hourly"]["cloud_cover_high"],
            "visibility": it["hourly"]["visibility"],
        } for it in data]
    raise RuntimeError(f"Open-Meteo chunk failed after {retries} retries: {last}")


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

    # --- AOD (Aerosol Optical Depth) fetch from Air Quality API ---
    AOD_API = "https://air-quality-api.open-meteo.com/v1/air-quality"
    AOD_LATS = [63.5 + i * 1.0 for i in range(4)]   # 63.5, 64.5, 65.5, 66.5
    AOD_LONS = [-24.0 + j * 2.0 for j in range(6)]  # -24, -22, -20, -18, -16, -14
    aod_grid_lats = [la for la in AOD_LATS for _ in AOD_LONS]
    aod_grid_lons = [lo for _ in AOD_LATS for lo in AOD_LONS]

    print(f"\nFetching AOD ({len(aod_grid_lats)} points)...", end="", flush=True)
    time.sleep(5)  # rate-limit courtesy delay after cloud fetch

    aod_points = []
    aod_times = []
    try:
        aod_params = {
            "latitude": ",".join(f"{x:.4f}" for x in aod_grid_lats),
            "longitude": ",".join(f"{x:.4f}" for x in aod_grid_lons),
            "hourly": "aerosol_optical_depth",
            "forecast_days": "5",
            "timezone": "UTC",
        }
        aod_resp = requests.get(AOD_API, params=aod_params, timeout=120)
        if aod_resp.status_code == 429:
            raise RuntimeError("Air Quality API rate limited (429)")
        aod_resp.raise_for_status()
        aod_data = aod_resp.json()
        if not isinstance(aod_data, list):
            aod_data = [aod_data]
        aod_times = aod_data[0].get("hourly", {}).get("time", []) if aod_data else []
        for it in aod_data:
            h = it.get("hourly", {})
            aod_points.append({
                "lat": it["latitude"],
                "lon": it["longitude"],
                "aod": h.get("aerosol_optical_depth", []),
            })
        print(f" ok ({len(aod_points)} points, {len(aod_times)} hours)")
    except Exception as e:
        print(f"\n  WARNING: AOD fetch failed ({e}); skipping AOD data.")
        aod_points = []
        aod_times = []

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
        "aod_times": aod_times,
        "aod_points": aod_points,
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
