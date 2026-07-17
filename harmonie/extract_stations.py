#!/usr/bin/env python3
"""
extract_stations.py — vedur STATION forecasts -> one compact JSON for the "W"
(weather-symbol) map layer of the Iceland Sky Predictor.

Mines the same data the public station-forecast table renders, but from the
machine-readable sources (NO image parsing):

  * the areas page          -> the shared time axis (wTime[] + interval[])
  * WeatherServlet f4grp     -> per-station forecast arrays, keyed by station id:
        W  weather symbol code (1..23, 0/'?' = none)   -> our own icon + label
        T  temperature (degC)        N  cloud cover (0..1 fraction)
        F  wind speed (m/s)          D2 wind direction (deg)
        R  precipitation (mm/interval)
        (grp 56 = ~26 "overview" stations; grp 121 = all stations)
  * wslinfo.js (VI.wsInfo)   -> station name + lat/lon + area

The weather descriptor set is fixed/standardised (VI.r.wtypeN); we draw our own
icons keyed on the integer W, so only the code stream is needed here.

Usage:  python extract_stations.py [--out FILE]
        default out = ../web/data/vedur_stations.json
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import re
import sys

import requests

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SunsetglowStationFetch/1.0"
# Primary: www.vedur.is (Icelandic site, reliable). Fallback: en.vedur.is (English, intermittent).
HOST_PRIMARY = "https://www.vedur.is"
HOST_FALLBACK = "https://en.vedur.is"
AREAS_PRIMARY = HOST_PRIMARY + "/vedur/spar/stadaspar"
AREAS_FALLBACK = HOST_FALLBACK + "/weather/forecasts/areas/"
SERVLET_PRIMARY = HOST_PRIMARY + "/WeatherServlet?op_x=f4grp&grp={grp}&local=1"
SERVLET_FALLBACK = HOST_FALLBACK + "/WeatherServlet?op_x=f4grp&grp={grp}&local=1"
WSLINFO_PRIMARY = HOST_PRIMARY + "/wstations/wslinfo.js"
WSLINFO_FALLBACK = HOST_FALLBACK + "/wstations/wslinfo.js"
GRP_OVERVIEW = 56          # whole-country overview (~26 stations)
GRP_ALL = 121              # every forecast station (hundreds)

# Keep the arrays that drive the icon + a useful tooltip. (TD/RT dropped to save size.)
KEEP = ("W", "T", "N", "F", "D2", "R")

# Standard vedur weather-type labels (VI.r.wtypeN) — embedded so the app and the
# data agree even offline. Index 0/'?' => no information.
WTYPE = {
    1: "Clear sky", 2: "Partly cloudy", 3: "Cloudy", 4: "Overcast",
    5: "Light rain", 6: "Rain", 7: "Light sleet", 8: "Sleet",
    9: "Light snow", 10: "Snow", 11: "Rain showers", 12: "Sleet shower",
    13: "Hail", 14: "Dust devil", 15: "Dust storm", 16: "Blowing snow",
    17: "Fog", 18: "Light drizzle", 19: "Drizzle", 20: "Freezing rain",
    21: "Hail", 22: "Light thunder", 23: "Thunder",
}

_DATE_RE = re.compile(r"new Date\((\d+),(\d+)-1,(\d+),(\d+),(\d+)\)")


def _get(url: str, retries: int = 3) -> str:
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=40)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            if attempt < retries:
                wait = 15
                print(f"  {type(e).__name__}, retry in {wait}s...", flush=True)
                import time; time.sleep(wait)
                continue
            raise
        if r.status_code in (500, 502, 503, 504):
            if attempt < retries:
                wait = 15
                print(f"  HTTP {r.status_code}, retry in {wait}s...", flush=True)
                import time; time.sleep(wait)
                continue
        r.raise_for_status()
        return r.text
    return ""  # unreachable


def _get_with_fallback(primary: str, fallback: str) -> str:
    """Try primary URL, fall back to alternate host on failure."""
    try:
        return _get(primary)
    except Exception as e:
        print(f"  Primary failed ({e}), trying fallback...", flush=True)
        return _get(fallback)


def _iso(y, mon, d, h, mi) -> str:
    # vedur emits new Date(Y, MON-1, D, h, mi); the literal first number IS the
    # human month. Times are GMT/UTC on the site (column header says GMT).
    return dt.datetime(int(y), int(mon), int(d), int(h), int(mi),
                       tzinfo=dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def parse_time_axis(areas_html: str):
    """Pull wTime[] (-> ISO list) and interval[] from the areas page."""
    m = re.search(r"wTime:\[(.*?)\]", areas_html, re.S)
    if not m:
        raise RuntimeError("wTime[] not found on areas page")
    times = [_iso(*g) for g in _DATE_RE.findall(m.group(1))]
    mi = re.search(r"interval:\[([0-9,\s]*)\]", areas_html, re.S)
    interval = [int(x) for x in mi.group(1).split(",") if x.strip()] if mi else []
    # issue time (tInfo modelInfo aTime) — best-effort, for display only
    issued = times[0] if times else None
    return times, interval, issued


def _nums(body: str):
    """'1,2,?,4.5,-3' -> [1, 2, None, 4.5, -3] (ints kept int, '?'/'' -> None)."""
    out = []
    for tok in body.split(","):
        tok = tok.strip()
        if tok == "" or tok == "?":
            out.append(None)
            continue
        try:
            f = float(tok)
            out.append(int(f) if f.is_integer() else round(f, 2))
        except ValueError:
            out.append(None)
    return out


# one station record:  6420:{'W':[...],'T':[...],...}
_STATION_RE = re.compile(r"(\d+):\{((?:'[A-Za-z0-9]+':\[[^\]]*\],?)+)\}")
_ARRAY_RE = re.compile(r"'([A-Za-z0-9]+)':\[([^\]]*)\]")


def parse_group(text: str):
    """WeatherServlet f4grp body -> { sid(str): {KEY: [..]} } for KEEP keys."""
    # restrict to the data:{...} object so the station regex can't catch the
    # outer 'time' field.
    di = text.find("'data':")
    body = text[di:] if di >= 0 else text
    stations = {}
    for sm in _STATION_RE.finditer(body):
        sid = sm.group(1)
        rec = {}
        for am in _ARRAY_RE.finditer(sm.group(2)):
            key = am.group(1)
            if key in KEEP:
                rec[key] = _nums(am.group(2))
        if rec.get("W"):
            stations[sid] = rec
    return stations


def parse_wslinfo(js: str):
    """VI.wsInfo = {sid:{'name':..,'lat':..,'lon':..,'area':..},...} -> dict."""
    m = re.search(r"VI\.wsInfo\s*=\s*\{(.*)\};", js, re.S)
    blob = m.group(1) if m else js
    info = {}
    rec_re = re.compile(
        r"(\d+):\{'name':'((?:[^'\\]|\\.)*)','lat':([\-0-9.]+),'lon':([\-0-9.]+)"
        r"(?:,'iTy':\d+)?(?:,'nst':\[[^\]]*\])?(?:,'area':'([^']*)')?")
    for m in rec_re.finditer(blob):
        sid, name, lat, lon, area = m.groups()
        info[sid] = {
            "name": name.replace("\\'", "'").replace("\\\\", "\\"),
            "lat": round(float(lat), 4), "lon": round(float(lon), 4),
            "area": area or "",
        }
    return info


def build(out_path: str) -> int:
    print("fetching areas page (time axis)…")
    times, interval, issued = parse_time_axis(
        _get_with_fallback(AREAS_PRIMARY, AREAS_FALLBACK))
    print(f"  {len(times)} time steps, issued {issued}")

    print("fetching station coords (wslinfo)…")
    wsinfo = parse_wslinfo(
        _get_with_fallback(WSLINFO_PRIMARY, WSLINFO_FALLBACK))
    print(f"  {len(wsinfo)} stations known")

    print("fetching overview group (56)…")
    overview = parse_group(
        _get_with_fallback(SERVLET_PRIMARY.format(grp=GRP_OVERVIEW),
                           SERVLET_FALLBACK.format(grp=GRP_OVERVIEW)))
    print(f"  {len(overview)} overview stations")

    print("fetching all-stations group (121)…")
    allst = parse_group(
        _get_with_fallback(SERVLET_PRIMARY.format(grp=GRP_ALL),
                           SERVLET_FALLBACK.format(grp=GRP_ALL)))
    print(f"  {len(allst)} total stations")

    # Union (all ⊇ overview); attach coords; tag overview tier.
    merged = dict(allst)
    for sid, rec in overview.items():
        merged.setdefault(sid, rec)
    stations = {}
    skipped = 0
    for sid, rec in merged.items():
        info = wsinfo.get(sid)
        if not info:
            skipped += 1
            continue
        stations[sid] = {
            "name": info["name"], "lat": info["lat"], "lon": info["lon"],
            "area": info["area"], **{k: rec[k] for k in KEEP if k in rec},
        }
    overview_ids = [s for s in overview if s in stations]
    print(f"  merged {len(stations)} stations ({skipped} had no coords), "
          f"{len(overview_ids)} in overview tier")

    doc = {
        "source": "Icelandic Meteorological Office (vedur.is) station forecasts",
        "fetched": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "issued": issued,
        "times": times,
        "interval": interval,
        "labels": WTYPE,
        "overview": overview_ids,
        "stations": stations,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"), ensure_ascii=False)
    size = os.path.getsize(out_path)
    print(f"wrote {out_path}  ({size/1024:.0f} KB, {len(stations)} stations, "
          f"{len(times)} steps)")
    return 0


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "..", "web", "data",
                                                   "vedur_stations.json"))
    a = ap.parse_args(argv)
    return build(os.path.abspath(a.out))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
