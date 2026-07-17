#!/usr/bin/env python3
"""extract_elements.py — vedur thattaspa wind/precip GIFs -> per-frame MAP OVERLAY
images (WebP) for the Iceland Sky Predictor, warped with the validated projective
(homography) georeference. No classification of unknown colours, no OCR, no API key.

WIND is shown AS-IS: its colour scale is dynamic per frame (gale-force adds colours
we can't know in advance), so we do NOT classify it — we warp the registered image
straight onto the map, keeping the speed colours AND the barbs (= direction).

PRECIP has a FIXED colour key (only the value labels rescale), so we keep only the
pixels that match the known precip palette (posterised to it) PLUS the red "L" /
blue "H" pressure markers, and make everything else (background, isobars, barbs,
coastline, text) transparent.

Each frame also carries its OWN legend strip, read verbatim from the image, so the
per-frame scale is always faithful.

Usage: python extract_elements.py [run] [--frames a,b,c] [--out FILE]
       run = YYMMDD_HHMM ; default = latest available.
"""
from __future__ import annotations
import base64
import datetime as dt
import io
import json
import re
import sys

import numpy as np
import requests
from PIL import Image
from scipy import ndimage

BASE = 'https://en.vedur.is/photos/'
PROD = {'wind': 'thattaspa_ig_island_10uv', 'precip': 'thattaspa_ig_island_urk-msl-10uv'}
# ECMWF (ecm-is) continuation products — vedur stitches these AFTER the 72 h HARMONIE
# window to reach ~7 days (coarser 3 h then 6 h cadence). Same base map + image
# dimensions (552x750) as HARMONIE, so the SAME warp homography + legend crop apply.
PROD_ECM = {'wind': 'thattaspa_ecm-is_island_10uv', 'precip': 'thattaspa_ecm-is_island_urk-msl-10uv'}
ELEM_PAGE = 'https://en.vedur.is/weather/forecasts/elements/'
HOST = 'https://en.vedur.is'

# Precip colour key is FIXED (only the value labels rescale). We match pixels to
# these to keep precip blobs (and, deliberately, the red L / blue H markers).
PRECIP_PAL = np.array([[254, 252, 131], [202, 211, 0], [170, 251, 120], [0, 170, 8],
                       [2, 92, 61], [6, 252, 237], [51, 102, 255], [0, 1, 124],
                       [119, 1, 97], [255, 20, 147], [255, 0, 0]])

# Projective georeference (homography) fitted to the thattaspa base map. RMS ~4.6 px.
L0, A0 = -19.0, 65.0
HP = [59.283907, 7.733016, 369.923796, -0.865369, -134.410065, 239.007669, 0.000179, 0.027392]
MAP_BOT = 523        # source rows >= this are the legend/footer, not the map

# Output overlay image (plate-carree over the map extent). Leaflet warps it onto
# the Mercator map between these bounds. ROW 0 = NORTH.
NORTH, SOUTH, WEST, EAST = 67.0, 62.9, -25.3, -12.7
IMG_W, IMG_H = 640, 420


def fetch_gif(prod, run, frame):
    url = '%s%s/%s_%03d.gif' % (BASE, prod, run, frame)
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


def latest_run(now=None):
    now = now or dt.datetime.utcnow()
    for back in range(0, 16):
        t = now - dt.timedelta(hours=3 * back)
        run = '%s_%02d00' % (t.strftime('%y%m%d'), (t.hour // 3) * 3)
        if fetch_gif(PROD['wind'], run, 1) is not None:
            return run
    return None


# ---- warp (uses the SAME validated homography as the original resample) ----
_XY = None


def _xy():
    """Source pixel (X, Y) for every output overlay pixel, via the homography.
    Row 0 = NORTH, last row = SOUTH. Cached (geometry is run-independent).

    Latitudes are spaced linearly in WEB-MERCATOR (not in latitude), so each overlay
    row lands where Leaflet actually draws it. Plate-carree row spacing made the warped
    map sit ~6-9 km too far NORTH over Iceland; Mercator spacing removes that shift."""
    global _XY
    if _XY is None:
        import math
        yN = math.log(math.tan(math.pi / 4 + math.radians(NORTH) / 2))
        yS = math.log(math.tan(math.pi / 4 + math.radians(SOUTH) / 2))
        ys = np.linspace(yN, yS, IMG_H)
        lats = np.degrees(2.0 * np.arctan(np.exp(ys)) - math.pi / 2)
        lons = np.linspace(WEST, EAST, IMG_W)
        LAT, LON = np.meshgrid(lats, lons, indexing='ij')
        dlon = LON - L0
        dlat = LAT - A0
        den = HP[6] * dlon + HP[7] * dlat + 1.0
        X = np.round((HP[0] * dlon + HP[1] * dlat + HP[2]) / den).astype(int)
        Y = np.round((HP[3] * dlon + HP[4] * dlat + HP[5]) / den).astype(int)
        _XY = (X, Y)
    return _XY


def _webp(rgba):
    buf = io.BytesIO()
    Image.fromarray(rgba, 'RGBA').save(buf, format='WEBP', lossless=True, method=4)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def _overlay_mask(a):
    """True where a pixel is a non-data line/glyph: low-saturation GRAY-to-DARK
    (isobars, coastline, wind barbs, black text/labels). Saturated colours (precip
    bands + the red L / blue H markers) and the bright white/tan background are
    kept. Dilated 1 px to catch anti-aliased edges."""
    flat = a.reshape(-1, 3).astype(int)
    bright = flat.mean(1)
    sat = flat.max(1) - flat.min(1)
    overlay = ((sat < 25) & (bright < 200)) | (bright < 45)
    return ndimage.binary_dilation(overlay.reshape(a.shape[:2]), iterations=4)


def _clean(a):
    """Remove the gray/black lines+text and fill the gap with the SURROUNDING colour
    by diffusion: iteratively average the known neighbours into the masked region.
    Unlike a nearest-pixel copy, this gives the true local colour (a label on a
    yellow blob fills yellow, not the nearest dark band) and leaves no gaps."""
    mask = _overlay_mask(a)
    if not mask.any():
        return a
    kb = ~mask
    src = a.astype(np.float32)
    f = src.copy()
    f[mask] = 0.0
    w = kb.astype(np.float32)
    for _ in range(40):
        f = ndimage.uniform_filter(f, size=(3, 3, 1), mode='nearest')
        w = ndimage.uniform_filter(w, size=3, mode='nearest')
        f[kb] = src[kb]
        w[kb] = 1.0
    out = f / np.maximum(w[..., None], 1e-6)
    out[kb] = src[kb]
    return np.clip(out, 0, 255).astype(np.uint8)


def warp_wind(a):
    """Wind AS-IS: warp the registered image; opaque over the map, transparent
    outside the domain / below the legend. Keeps speed colours + direction barbs."""
    X, Y = _xy()
    Hs, Ws = a.shape[:2]
    valid = (X >= 0) & (X < Ws) & (Y >= 0) & (Y < min(Hs, MAP_BOT))
    rgb = a[np.clip(Y, 0, Hs - 1), np.clip(X, 0, Ws - 1)]
    rgba = np.dstack([rgb, np.where(valid, 255, 0).astype(np.uint8)])
    return _webp(rgba)


def warp_precip(a):
    """Precip with the FIXED palette: first CLEAN the gray/black isobars, coastline,
    barbs and text (inpaint from surroundings) so they don't leave lines/gaps, then
    keep saturated pixels matching a known precip colour (posterised) PLUS the red L
    / blue H markers; everything else (background) -> transparent."""
    a = _clean(a)
    X, Y = _xy()
    Hs, Ws = a.shape[:2]
    valid = (X >= 0) & (X < Ws) & (Y >= 0) & (Y < min(Hs, MAP_BOT))
    rgb = a[np.clip(Y, 0, Hs - 1), np.clip(X, 0, Ws - 1)].astype(int)
    sat = rgb.max(2) - rgb.min(2)
    d = ((rgb[:, :, None, :] - PRECIP_PAL[None, None, :, :]) ** 2).sum(3)
    bi = d.argmin(2)
    near = np.sqrt(d.min(2))
    keep = valid & (sat > 50) & (near < 120)
    red = bi == (len(PRECIP_PAL) - 1)             # the red band = 50 mm precip + the "L" marker
    bi = ndimage.median_filter(bi, size=5)        # strong despeckle -> seamless blobs
    bi[red] = len(PRECIP_PAL) - 1                 # but keep the red "L" intact
    rgba = np.dstack([PRECIP_PAL[bi], np.where(keep, 255, 0)]).astype(np.uint8)
    return _webp(rgba)


# ---- per-frame legend strip (read verbatim from the image) ----
def _legend_region(a: np.ndarray):
    """Locate the legend colorbar: (y, x0, x1) of the horizontal band of SOLID,
    saturated colour cells below the map (vedur draws the key at the bottom). Works
    for any colour scheme, so the dynamic wind scale is read straight from pixels."""
    H, W, _ = a.shape
    x_lo, x_hi = 110, W - 5

    def solid_sat(y):
        strip = a[max(0, y - 3):y + 4, x_lo:x_hi].astype(int)
        vstd = strip.std(0).mean(1)                       # low where the bar is a solid colour
        seg = a[y, x_lo:x_hi].astype(int)
        sat = seg.max(1) - seg.min(1)
        return (vstd < 16) & (sat > 14)

    ys = range(MAP_BOT + 1, min(H - 2, MAP_BOT + 13))
    y = max(ys, key=lambda yy: int(solid_sat(yy).sum()))
    mask = solid_sat(y)
    xs = [x_lo + i for i, m in enumerate(mask) if m]
    if not xs:
        return None
    runs, s, p = [], xs[0], xs[0]
    for x in xs[1:]:
        if x - p <= 3:
            p = x
        else:
            runs.append((s, p)); s = x; p = x
    runs.append((s, p))
    x0, x1 = max(runs, key=lambda r: r[1] - r[0])
    return y, x0, x1


def extract_legend(a: np.ndarray):
    """base64-PNG of the untampered legend strip (colorbar + numeric labels), so
    each frame's own scale is shown verbatim on our map — no OCR, no guessing."""
    reg = _legend_region(a)
    if reg is None:
        return None
    y, x0, x1 = reg
    # Vertical crop = top of the colorbar band THROUGH the bottom of the numeric
    # labels (detected, never chopped). Labels sit a few px below the band after a
    # small blank gap; stop once that one text line ends so we don't grab the
    # footer/date line further down.
    Himg, Wimg = a.shape[:2]

    def band_row(yy):
        seg = a[yy, x0:x1].astype(int)
        return (seg.max(1) - seg.min(1)).mean() > 40 and seg.mean() > 110

    top = y
    while top - 1 > MAP_BOT and band_row(top - 1):
        top -= 1
    bot = y
    while bot + 1 < Himg and band_row(bot + 1):
        bot += 1
    # The value-label line sits a few px below the colour bar (after a small blank gap). Use a
    # FIXED extent below the bar rather than an adaptive text scan: the scan was fragile — when
    # the bar's bottom edge landed just outside the detected band it was mistaken for the label
    # line and the blank gap then cut the crop short (bar only, NO numbers — the precip-legend
    # bug). The label line is always within ~12 px of the bar bottom and nothing else lives in
    # the bar's x-span below it, so a fixed extent reliably grabs the bar + its numbers.
    label_bot = min(Himg - 2, bot + 13)
    crop = Image.fromarray(a[max(MAP_BOT + 1, top - 1):label_bot + 1,
                             max(0, x0 - 9):min(Wimg, x1 + 9)])
    buf = io.BytesIO()
    crop.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def run_valid_time(run, frame):
    base = dt.datetime.strptime(run, '%y%m%d_%H%M').replace(tzinfo=dt.timezone.utc)
    return base + dt.timedelta(hours=frame)


def build(run, frames):
    overlays = {'wind': [], 'precip': []}
    legends = {'wind': [], 'precip': []}
    times, kept = [], []
    for fr in frames:
        gifs = {k: fetch_gif(PROD[k], run, fr) for k in PROD}
        if gifs['wind'] is None:
            continue
        kept.append(fr)
        times.append(run_valid_time(run, fr).strftime('%Y-%m-%dT%H:%MZ'))
        for k in PROD:
            a = gifs[k]
            if a is None:
                overlays[k].append(None)
                legends[k].append(None)
                continue
            overlays[k].append(warp_wind(a) if k == 'wind' else warp_precip(a))
            legends[k].append(extract_legend(a))
    if not kept:
        raise SystemExit('run %s: no frames available' % run)
    return {
        'source': 'vedur_thattaspa_ig', 'run': run,
        'generated': dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'north': NORTH, 'south': SOUTH, 'west': WEST, 'east': EAST,
        'img_w': IMG_W, 'img_h': IMG_H, 'fmt': 'webp',
        'frames': kept, 'times': times,
        'wind': overlays['wind'], 'precip': overlays['precip'],
        'wind_legends': legends['wind'], 'precip_legends': legends['precip'],
    }


def fetch_gif_abs(url):
    """Fetch a GIF by absolute URL -> RGB array (or None)."""
    try:
        r = requests.get(url, timeout=45)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    return np.array(Image.open(io.BytesIO(r.content)).convert('RGB'))


def ecmwf_continuation(after_dt):
    """Parse vedur's elements page for the ECMWF (ecm-is) WIND frames whose valid
    time is AFTER the HARMONIE window, deriving the matching precip URL by product
    substitution so wind+precip stay on ONE shared time grid. Returns a sorted list
    of (valid_dt, wind_url, precip_url). Authoritative (uses vedur's own run + frame
    selection + 3h/6h cadence); empty on any failure so the 72 h product is intact."""
    try:
        html = requests.get(ELEM_PAGE, timeout=45).text
    except Exception:
        return []
    m = re.search(r"VI\.imgConf\[0\]\s*=\s*\{(.*?)\};", html, re.S)
    if not m:
        return []
    out, seen = [], set()
    for _, url in re.findall(r"'(\d+)'\s*:\s*'([^']*\.gif)'", m.group(1)):
        if 'ecm-is' not in url:
            continue
        mm = re.search(r'(\d{6})_(\d{4})_(\d+)\.gif$', url)
        if not mm:
            continue
        run = mm.group(1) + '_' + mm.group(2)
        valid = run_valid_time(run, int(mm.group(3)))
        if valid <= after_dt:
            continue
        key = valid.strftime('%Y%m%d%H')
        if key in seen:
            continue
        seen.add(key)
        wind_url = HOST + url
        precip_url = HOST + url.replace('_island_10uv', '_island_urk-msl-10uv')
        out.append((valid, wind_url, precip_url))
    out.sort(key=lambda x: x[0])
    return out


def append_ecmwf(res):
    """APPEND the ECMWF 7-day continuation to a built (72 h HARMONIE) result, in
    place. Additive only — the 72 h hourly product is never modified. A frame is
    appended only when BOTH wind and precip warp successfully, so the two layers
    stay on one shared time grid (the app indexes them together)."""
    if not res.get('times'):
        return res
    last = dt.datetime.strptime(res['times'][-1], '%Y-%m-%dT%H:%MZ').replace(tzinfo=dt.timezone.utc)
    cont = ecmwf_continuation(last)
    added = 0
    for valid, wurl, purl in cont:
        w = fetch_gif_abs(wurl)
        p = fetch_gif_abs(purl)
        if w is None or p is None:
            continue
        res['wind'].append(warp_wind(w))
        res['precip'].append(warp_precip(p))
        res['wind_legends'].append(extract_legend(w))
        res['precip_legends'].append(extract_legend(p))
        res['times'].append(valid.strftime('%Y-%m-%dT%H:%MZ'))
        res['frames'].append((res['frames'][-1] if res['frames'] else 0) + 1)
        added += 1
    res['ecmwf_frames'] = added
    return res


def main(argv):
    run, frames, out = None, list(range(1, 73)), 'elements_latest.json'
    no_ecmwf = False
    i = 0
    while i < len(argv):
        if argv[i] == '--frames':
            frames = [int(x) for x in argv[i + 1].split(',')]; i += 2
        elif argv[i] == '--out':
            out = argv[i + 1]; i += 2
        elif argv[i] == '--no-ecmwf':       # 72 h HARMONIE only (skip the 7-day continuation)
            no_ecmwf = True; i += 1
        else:
            run = argv[i]; i += 1
    if run is None:
        run = latest_run()
        if run is None:
            raise SystemExit('no thattaspa run reachable')
    print('run=%s n_frames=%d' % (run, len(frames)))
    res = build(run, frames)
    n72 = len(res['frames'])
    if not no_ecmwf:
        # Additive: append the ECMWF 7-day continuation. Guard so any failure leaves
        # the 72 h HARMONIE product fully intact (the existing high-res workflow).
        try:
            append_ecmwf(res)
        except Exception as e:
            print('ECMWF continuation skipped: %s' % e)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(res, f, separators=(',', ':'))
    import gzip
    import os
    gz = len(gzip.compress(open(out, 'rb').read(), 6)) / 1024 / 1024
    print('wrote %s  (%d frames = %d HARMONIE + %d ECMWF, %dx%d %s, %.1f MB / %.1f MB gz)'
          % (out, len(res['frames']), n72, res.get('ecmwf_frames', 0), res['img_w'], res['img_h'],
             res['fmt'], os.path.getsize(out) / 1024 / 1024, gz))


if __name__ == '__main__':
    main(sys.argv[1:])
