#!/usr/bin/env python3
"""
extract.py — vedur HARMONIE 2.5 km cloud PNGs -> compact JSON cloud grids for the
Iceland Sky Predictor.

Runs anywhere with internet (designed for GitHub Actions): downloads the
harmonie_island low/mid/high cloud PNGs, decodes the posterized okta colors,
masks the coastline/town-dots/labels and nearest-fills them, self-calibrates the
pixel->lat/lon mapping against Iceland's coastline, resamples onto a regular
lat/lon grid, and writes one JSON per run. No API key, no rate limit, no
per-point billing.

Usage:  python extract.py [run] [--frames a,b,c] [--out FILE]
        run = YYMMDD_HHMM (e.g. 260605_0600); default = latest available.
"""
from __future__ import annotations
import datetime as dt
import io
import json
import sys
import time

import numpy as np
import requests
from PIL import Image
from scipy import ndimage

BASE = 'https://en.vedur.is/photos/'
LAYER_FILE = {'low': 'lcc', 'mid': 'mcc', 'high': 'hcc'}
MAP_BOT = 377                       # map rows 0..376; rows 378-385 are the legend
                                    # colorbar (verified lcc/mcc/hcc). 341 cropped
                                    # ~36 rows of real map (ocean south of Iceland).

# Posterized cloud palettes — each vedur layer uses a DIFFERENT colormap
# (verified against the source PNGs: low=blue, mid=red, high=orange ramps).
# Colors are the 4 posterized map colors in clear -> overcast order.
PAL_BY_LAYER = {
    'low':  np.array([[254, 254, 254], [153, 204, 255], [0, 204, 255], [50, 101, 254]]),
    'mid':  np.array([[254, 254, 254], [255, 204, 153], [254, 104, 92], [169, 50, 21]]),
    'high': np.array([[254, 254, 254], [254, 254, 0], [254, 169, 0], [254, 114, 0]]),
}
OKTA = np.array([0, 2, 5, 8])      # band centers on the 0..8 okta scale (same for all layers)

# Georeference: affine fit (lat,lon)->source pixel (x,y), calibrated against 8
# town control points (RMS 3.7 px ~= 4 km, max 5 px). Negligible rotation.
CX = (42.6812, 1.0820, 1007.7625)      # x = CX0*lon + CX1*lat + CX2
CY = (-0.6474, -100.1357, 6671.7858)   # y = CY0*lon + CY1*lat + CY2
NODATA = 255                            # output okta value for cells outside the source map

# Output grid (Iceland-focused, regular lat/lon). step ~0.05 deg ~= 5.5 km.
OUT = dict(lat0=62.9, lat1=67.0, lon0=-25.3, lon1=-12.7, step=0.05)


def fetch_png(vfile: str, run: str, frame: int):
    url = '%sharmonie_island_%s/%s_%d.png' % (BASE, vfile, run, frame)
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=45)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < 3:
                time.sleep(15)
                continue
            return None
        if r.status_code in (500, 502, 503, 504) and attempt < 3:
            time.sleep(15)
            continue
        if r.status_code != 200:
            return None
        return np.array(Image.open(io.BytesIO(r.content)).convert('RGB'))


def latest_run(now: dt.datetime | None = None):
    """Find the most recent published run (vedur cycles every 3 h: 00/03/06/.../21
    UTC), probing lcc frame 6 to confirm the run actually has data."""
    now = now or dt.datetime.utcnow()
    for back in range(0, 16):                      # up to 45 h back, every 3 h
        t = now - dt.timedelta(hours=3 * back)
        run = '%s_%02d00' % (t.strftime('%y%m%d'), (t.hour // 3) * 3)
        if fetch_png('lcc', run, 6) is not None:
            return run
    return None


def _overlay_mask(a: np.ndarray) -> np.ndarray:
    """True where the pixel is a non-cloud overlay (coastline / town-dot / text /
    border / graticule): gray-ish or very dark. Dilated 1px to catch anti-alias
    edges. Kept tight — the majority-vote resample ignores these, so we must NOT
    over-mask real cloud."""
    H, W, _ = a.shape
    flat = a.reshape(-1, 3)
    bright = flat.mean(1)
    isgray = (np.abs(flat[:, 0] - flat[:, 1]) < 24) & (np.abs(flat[:, 1] - flat[:, 2]) < 24)
    overlay = ((isgray & (bright < 245)) | (bright < 55)).reshape(H, W)
    return ndimage.binary_dilation(overlay, iterations=1)


def classify(a_full: np.ndarray, pal: np.ndarray):
    """Cropped map -> (per-pixel okta grid using the layer palette, line mask)."""
    a = a_full[:MAP_BOT].astype(int)
    H, W, _ = a.shape
    flat = a.reshape(-1, 3)
    idx = ((flat[:, None, :] - pal[None, :, :]) ** 2).sum(2).argmin(1)
    okt = OKTA[idx].reshape(H, W)
    return okt, _overlay_mask(a)


def extract_okta(a_full: np.ndarray, pal: np.ndarray) -> np.ndarray:
    """Per-pixel okta classification (no fill) — for validation/visualization."""
    return classify(a_full, pal)[0]


def calibrate_geo(a_full=None):
    """Deprecated: georeference is now a fixed affine (see CX/CY)."""
    return None


def resample(okt, overlay, out_lats, out_lons):
    """Map each output (lat,lon) cell to the source via the affine fit and take the
    MAJORITY okta among the REAL-cloud (non-overlay) source pixels in that cell's
    footprint. Robust to coastline/graticule lines without eating cloud. Cells with
    no valid cloud pixels (outside the map, or fully under overlay) -> NODATA.

    Output cell footprint in source px ~ (5 lat, 3 lon), so we vote over a (5,3)
    window of the per-okta-level coverage."""
    good = (~overlay).astype(np.float32)
    LAT, LON = np.meshgrid(out_lats, out_lons, indexing='ij')
    X = np.round(CX[0] * LON + CX[1] * LAT + CX[2]).astype(int)
    Y = np.round(CY[0] * LON + CY[1] * LAT + CY[2]).astype(int)
    Hs, Ws = okt.shape
    valid = (X >= 0) & (X < Ws) & (Y >= 0) & (Y < Hs)
    Yc, Xc = np.clip(Y, 0, Hs - 1), np.clip(X, 0, Ws - 1)
    win = (5, 3)
    best = np.zeros(X.shape, np.float32)
    out = np.full(X.shape, NODATA, int)
    tot = np.zeros(X.shape, np.float32)
    for lev in OKTA:
        cov = ndimage.uniform_filter(((okt == lev) & (~overlay)).astype(np.float32),
                                     size=win, mode='constant')[Yc, Xc]
        tot += cov
        upd = cov > best
        best = np.where(upd, cov, best)
        out = np.where(upd, lev, out)
    # Cells whose footprint is ENTIRELY under overlay (no cloud pixels to vote on)
    # fall back to the nearest real-cloud pixel — avoids holes without eating cloud.
    ind = ndimage.distance_transform_edt(overlay, return_distances=False, return_indices=True)
    nearest = okt[tuple(ind)][Yc, Xc]
    out = np.where(tot > 0, out, nearest)
    return np.where(valid, out, NODATA)


def run_valid_time(run: str, frame: int) -> dt.datetime:
    base = dt.datetime.strptime(run, '%y%m%d_%H%M').replace(tzinfo=dt.timezone.utc)
    return base + dt.timedelta(hours=frame)


def build(run: str, frames):
    out_lats = np.arange(OUT['lat1'], OUT['lat0'] - 1e-9, -OUT['step'])   # N->S
    out_lons = np.arange(OUT['lon0'], OUT['lon1'] + 1e-9, OUT['step'])    # W->E
    data = {'low': [], 'mid': [], 'high': []}
    times = []
    kept = []
    for fr in frames:
        pngs = {layer: fetch_png(vfile, run, fr) for layer, vfile in LAYER_FILE.items()}
        if pngs['low'] is None:
            continue                       # this run hasn't published this frame yet
        kept.append(fr)
        times.append(run_valid_time(run, fr).strftime('%Y-%m-%dT%H:%MZ'))
        for layer in LAYER_FILE:
            a = pngs[layer]
            if a is None:
                grid = np.full((len(out_lats), len(out_lons)), NODATA, int)
            else:
                okt, overlay = classify(a, PAL_BY_LAYER[layer])
                grid = resample(okt, overlay, out_lats, out_lons)
            data[layer].append(grid.astype(np.uint8).tolist())
    if not kept:
        raise SystemExit('run %s: no frames available' % run)
    return {
        'source': 'vedur_harmonie_2p5km',
        'scale': 'okta_0_8',
        'nodata': NODATA,
        'run': run,
        'generated': dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'lat0': round(float(out_lats[0]), 4), 'lat1': round(float(out_lats[-1]), 4),
        'lon0': round(float(out_lons[0]), 4), 'lon1': round(float(out_lons[-1]), 4),
        'nlat': len(out_lats), 'nlon': len(out_lons), 'step': OUT['step'],
        'frames': kept, 'times': times,
        'low': data['low'], 'mid': data['mid'], 'high': data['high'],
    }


def main(argv):
    run = None
    frames = list(range(1, 79))      # default: +1..+78 h hourly (full HARMONIE horizon)
    out = 'harmonie_latest.json'
    i = 0
    while i < len(argv):
        if argv[i] == '--frames':
            frames = [int(x) for x in argv[i + 1].split(',')]; i += 2
        elif argv[i] == '--out':
            out = argv[i + 1]; i += 2
        else:
            run = argv[i]; i += 1
    if run is None:
        run = latest_run()
        if run is None:
            raise SystemExit('no HARMONIE run reachable')
    print('run=%s n_frames=%d' % (run, len(frames)))
    res = build(run, frames)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(res, f, separators=(',', ':'))
    import gzip
    import os
    raw = open(out, 'rb').read()
    gz_kb = len(gzip.compress(raw, 6)) / 1024
    nd = sum(1 for fr in res['low'] for row in fr for v in row if v == 255)
    tot = len(res['low']) * res['nlat'] * res['nlon']
    print('wrote %s  (%d frames, %dx%d grid, %.0f KB raw / %.0f KB gzip, no-data %.0f%%)'
          % (out, len(res['frames']), res['nlat'], res['nlon'],
             os.path.getsize(out) / 1024, gz_kb, 100 * nd / max(1, tot)))


if __name__ == '__main__':
    main(sys.argv[1:])
