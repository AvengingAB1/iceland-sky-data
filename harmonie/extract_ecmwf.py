#!/usr/bin/env python3
"""
extract_ecmwf.py - ECMWF IFS 9 km (O1280) long-range fields for the Sunsetglow app, read from
Open-Meteo's open data bucket on AWS (s3://openmeteo/data_spatial/ecmwf_ifs, anonymous, CC BY 4.0).
No Open-Meteo API calls are used, so there is no daily call budget.

Outputs (same schemas the app already reads for MEPS):
  ecmwf_latest.json           cloud grid: low/mid/high in 5 % steps (div 20) on a regular lat/lon
                              grid (nearest O1280 point, values exact) + visibility `vis`
                              (base64 bytes per frame, value*50 m, 101 = clearer than 5 km).
  ecmwf_elements_latest.json  10 m wind + precipitation-rate WebP overlays (MEPS colour bands).

Steps kept: 3-hourly to +144 h, then 6-hourly to +360 h (00Z/12Z runs reach +360 h).
`precipitation` in the 3/6-hourly files is the TOTAL over the preceding interval (verified against
the Open-Meteo API, which spreads it evenly per hour), so the overlay shows total / interval = mm/h.

Usage: python extract_ecmwf.py --region nnorway|iceland [--out F] [--elements-out F] [--run YYYY-MM-DDTHH]
"""
from __future__ import annotations
import argparse
import base64
import datetime as dt
import io
import json
import math
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import fsspec
from omfiles import OmFileReader
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import RegularGridInterpolator

S3_HTTP = 'https://openmeteo.s3.amazonaws.com'
PREFIX = 'data_spatial/ecmwf_ifs'
REGIONS = {
    # box = the app's long-range grid box (region.js `grid`); dlon keeps cells ~square.
    'nnorway': dict(lat0=64.6, lat1=72.5, lon0=4.8, lon1=28.0, dlat=0.07, dlon=0.2, img=(520, 500)),
    'iceland': dict(lat0=63.0, lat1=69.9, lon0=-25.2, lon1=-12.6, dlat=0.07, dlon=0.16, img=(470, 520)),
}
DIV, NODATA = 20, 255
VIS_STEP, VIS_CLEAR = 50, 101
VARS = ('cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high', 'visibility',
        'precipitation', 'wind_u_component_10m', 'wind_v_component_10m')
ALPHA = 190
WIND_EDGES = [0, 3, 5, 8, 11, 14, 17, 21, 25, 29]
WIND_RGB = [(70, 110, 170), (60, 150, 190), (70, 180, 140), (120, 200, 90), (220, 220, 70),
            (240, 170, 50), (235, 110, 40), (215, 50, 50), (170, 30, 110), (120, 40, 160)]
PRECIP_EDGES = [0.1, 0.3, 0.5, 1, 2, 5, 10]
PRECIP_RGB = [(170, 210, 245), (120, 180, 240), (70, 140, 230), (40, 95, 205),
              (60, 50, 180), (140, 40, 170), (210, 40, 120)]
CREDIT = 'ECMWF IFS 9 km (CC BY 4.0) via Open-Meteo open data on AWS'


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode('ascii')


# ---------- O1280 octahedral reduced Gaussian grid ----------
def o1280():
    n = 1280
    x, _ = np.polynomial.legendre.leggauss(2 * n)
    lats = np.degrees(np.arcsin(x))[::-1]                                  # north -> south
    npts = np.array([20 + 4 * i for i in range(n)] + [20 + 4 * i for i in reversed(range(n))])
    start = np.concatenate([[0], np.cumsum(npts)[:-1]])
    return lats, npts, start


class BlockFS:
    """fsspec stand-in for OmFileReader: it reads many small ranges via cat_file(path, start, end);
    serving them from cached aligned 512 KB blocks cuts ~1000 requests per file to ~10."""
    def __init__(self, fs, bs=512 * 1024):
        self.fs, self.bs, self.blocks, self.lock = fs, bs, {}, threading.Lock()

    def size(self, path):
        return self.fs.size(path)

    def _block(self, path, k):
        with self.lock:
            if (path, k) in self.blocks:
                return self.blocks[(path, k)]
        for attempt in range(4):
            try:
                data = self.fs.cat_file(path, start=k * self.bs, end=(k + 1) * self.bs)
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(3 * (attempt + 1))
        with self.lock:
            self.blocks[(path, k)] = data
        return data

    def cat_file(self, path, start=None, end=None, **kw):
        start = start or 0
        end = end if end is not None else self.size(path)
        k0, k1 = start // self.bs, (end - 1) // self.bs
        buf = b''.join(self._block(path, k) for k in range(k0, k1 + 1))
        return buf[start - k0 * self.bs: end - k0 * self.bs]

    def __getattr__(self, n):
        return getattr(self.fs, n)


def get_json(url):
    return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'sunsetglow-data'}), timeout=60).read())


def list_run_files(run):
    """Valid times (as 'YYYY-MM-DDTHHMM') of the .om files in a run directory."""
    p = '%s/%s/%s/' % (PREFIX, run.strftime('%Y/%m/%d'), run.strftime('%H00Z'))
    keys, token = [], None
    while True:
        u = '%s/?list-type=2&prefix=%s&max-keys=1000' % (S3_HTTP, p) + ('&continuation-token=%s' % urllib.parse.quote(token) if token else '')
        t = urllib.request.urlopen(u, timeout=60).read().decode()
        keys += re.findall(r'<Key>[^<]*/(\d{4}-\d{2}-\d{2}T\d{4})\.om</Key>', t)
        m = re.search(r'<NextContinuationToken>([^<]+)</NextContinuationToken>', t)
        if not m:
            break
        token = m.group(1)
    return p, sorted(keys)


def pick_run(explicit=None):
    """Newest COMPLETE 00Z/12Z run (those reach +360 h); 06Z/18Z stop at +144 h."""
    if explicit:
        run = dt.datetime.strptime(explicit, '%Y-%m-%dT%H').replace(tzinfo=dt.timezone.utc)
        return (run,) + list_run_files(run)
    ref = dt.datetime.fromisoformat(get_json('%s/%s/latest.json' % (S3_HTTP, PREFIX))['reference_time'].replace('Z', '+00:00'))
    for back in range(0, 5):
        run = ref - dt.timedelta(hours=6 * back)
        if run.hour not in (0, 12):
            continue
        p, files = list_run_files(run)
        last = (run + dt.timedelta(hours=360)).strftime('%Y-%m-%dT%H%M')
        if files and files[-1] == last:
            return run, p, files
        print('  run %s incomplete (%d files, last %s)' % (run.strftime('%Y-%m-%d %HZ'), len(files), files[-1] if files else '-'))
    raise SystemExit('no complete ECMWF 00Z/12Z run found')


def keep_steps(run, files):
    out = []
    for f in files:
        vt = dt.datetime.strptime(f, '%Y-%m-%dT%H%M').replace(tzinfo=dt.timezone.utc)
        h = int(round((vt - run).total_seconds() / 3600))
        if (h <= 144 and h % 3 == 0) or (h > 144 and h % 6 == 0):
            out.append((h, vt, f))
    return out


def build(region, run, prefix, files):
    R = REGIONS[region]
    lats, npts, start = o1280()
    rows = np.where((lats >= R['lat0'] - 0.15) & (lats <= R['lat1'] + 0.15))[0]
    a, b = int(start[rows[0]]), int(start[rows[-1]] + npts[rows[-1]])
    # source point coordinates within the contiguous latitude band [a, b)
    plat = np.concatenate([np.full(npts[i], lats[i]) for i in rows])
    plon = np.concatenate([np.arange(npts[i]) * 360.0 / npts[i] for i in rows])
    plon = np.where(plon > 180, plon - 360, plon)
    # regular output grid, row 0 = north
    out_lats = np.round(np.arange(R['lat1'], R['lat0'] - 1e-9, -R['dlat']), 5)
    out_lons = np.round(np.arange(R['lon0'], R['lon1'] + 1e-9, R['dlon']), 5)
    # nearest source point per output cell: exact row (nearest Gaussian latitude), then nearest
    # longitude on that row (the grid is regular within a row) -> values identical to the source.
    ri = np.array([rows[np.argmin(np.abs(lats[rows] - la))] for la in out_lats])
    idx = np.empty((len(out_lats), len(out_lons)), np.int64)
    for r, i in enumerate(ri):
        n = npts[i]
        j = np.rint((out_lons % 360) / (360.0 / n)).astype(np.int64) % n
        idx[r] = (start[i] + j) - a
    steps = keep_steps(run, files)
    print('run %s: %d steps (+%d..+%d h), band %d points, grid %dx%d' % (
        run.strftime('%Y-%m-%d %HZ'), len(steps), steps[0][0], steps[-1][0], b - a, len(out_lats), len(out_lons)))

    def read_step(s):
        h, vt, f = s
        fs = BlockFS(fsspec.filesystem('s3', anon=True, skip_instance_cache=True))
        r = OmFileReader.from_fsspec(fs, 'openmeteo/%s%s.om' % (prefix, f))
        vals = {}
        for v in VARS:
            try:
                child = r.get_child_by_name(v)
            except ValueError:                      # e.g. no precipitation in the +0 h analysis
                vals[v] = np.full(idx.shape, np.nan, 'float32')
                continue
            band = np.asarray(child.read_array((slice(0, 1), slice(a, b)))[0], dtype='float32')
            vals[v] = band[idx]
        return h, vt, vals

    t = time.time()
    with ThreadPoolExecutor(8) as ex:
        got = list(ex.map(read_step, steps))
    print('  read %d files x %d vars in %.0fs' % (len(got), len(VARS), time.time() - t))
    hours = [h for h, _, _ in got]
    times = [vt.strftime('%Y-%m-%dT%H:%MZ') for _, vt, _ in got]

    def q_cloud(x):
        return np.where(np.isnan(x), NODATA, np.clip(np.rint(x / 100.0 * DIV), 0, DIV)).astype(np.uint8)

    def q_vis(x):
        return np.where(np.isnan(x), NODATA, np.minimum(np.rint(x / VIS_STEP), VIS_CLEAR)).astype(np.uint8)

    grid = {
        'source': 'ecmwf_ifs_9km', 'region': region,
        'scale': 'fraction_div', 'div': DIV, 'nodata': NODATA,
        'run': run.strftime('%y%m%d_%H%M'),
        'generated': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'lat0': float(out_lats[0]), 'lat1': float(out_lats[-1]),
        'lon0': float(out_lons[0]), 'lon1': float(out_lons[-1]),
        'nlat': len(out_lats), 'nlon': len(out_lons), 'dlat': R['dlat'], 'dlon': R['dlon'],
        'frames': hours, 'times': times,
        # one base64 string of nlat*nlon bytes per frame (row 0 = north): ~1 byte/cell on phones
        'cloud_enc': 'u8-b64',
        'low': [b64(q_cloud(v['cloud_cover_low'])) for _, _, v in got],
        'mid': [b64(q_cloud(v['cloud_cover_mid'])) for _, _, v in got],
        'high': [b64(q_cloud(v['cloud_cover_high'])) for _, _, v in got],
        'vis': [b64(q_vis(v['visibility'])) for _, _, v in got],
        'vis_enc': 'u8-b64', 'vis_step': VIS_STEP, 'vis_clear': VIS_CLEAR,
        'credit': CREDIT,
    }
    elements = render_elements(region, run, got, out_lats, out_lons)
    return grid, elements


# ---------- wind / precip overlays (same look and schema as the MEPS elements feed) ----------
def colourise(v, edges, rgb, transparent_below=True):
    i = np.digitize(v, edges) - 1
    out = np.zeros(v.shape + (4,), np.uint8)
    ok = i >= 0 if transparent_below else np.ones(v.shape, bool)
    out[..., :3] = np.array(rgb, np.uint8)[np.clip(i, 0, len(rgb) - 1)]
    out[..., 3] = np.where(ok & ~np.isnan(v), ALPHA, 0)
    return out


def draw_arrows(img, spd, dirn, step=40):
    d = ImageDraw.Draw(img)
    H, W = spd.shape
    for r in range(step // 2, H, step):
        for c in range(step // 2, W, step):
            s, frm = spd[r, c], dirn[r, c]
            if np.isnan(s) or np.isnan(frm) or s < 1:
                continue
            to = math.radians(frm + 180.0)
            L = 9 + min(s, 25) * 0.35
            dx, dy = math.sin(to) * L, -math.cos(to) * L
            x0, y0, x1, y1 = c - dx / 2, r - dy / 2, c + dx / 2, r + dy / 2
            d.line([(x0, y0), (x1, y1)], fill=(20, 20, 30, 230), width=2)
            for ang in (150, -150):
                ha = to + math.radians(ang)
                d.line([(x1, y1), (x1 + math.sin(ha) * 4, y1 - math.cos(ha) * 4)], fill=(20, 20, 30, 230), width=2)


def webp_b64(img):
    buf = io.BytesIO()
    img.save(buf, format='WEBP', lossless=True, method=4)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def legend_b64(edges, rgb, fmt):
    cell, h = 34, 14
    img = Image.new('RGBA', (cell * len(rgb), h * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for i, col in enumerate(rgb):
        d.rectangle([i * cell, 0, (i + 1) * cell - 1, h - 1], fill=col + (255,))
        d.text((i * cell + 2, h + 1), fmt(edges[i]), fill=(25, 25, 25, 255), font=font)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def render_elements(region, run, got, out_lats, out_lons):
    R = REGIONS[region]
    W, H = R['img']
    north, south, west, east = R['lat1'], R['lat0'], R['lon0'], R['lon1']
    ym = lambda la: math.log(math.tan(math.pi / 4 + math.radians(la) / 2))
    yN, yS = ym(north), ym(south)
    plat = np.degrees(np.arctan(np.sinh(np.array([yN - (r + 0.5) / H * (yN - yS) for r in range(H)]))))
    plon = west + (np.arange(W) + 0.5) / W * (east - west)
    P = np.stack(np.meshgrid(plat, plon, indexing='ij'), -1).reshape(-1, 2)
    la_asc = out_lats[::-1]

    def interp(field):                       # bilinear on the regular grid -> smooth overlay
        f = RegularGridInterpolator((la_asc, out_lons), field[::-1], bounds_error=False, fill_value=np.nan)
        return f(P).reshape(H, W)

    wind, precip, times, frames = [], [], [], []
    prev_h = None
    for h, vt, v in got:
        u, w = v['wind_u_component_10m'], v['wind_v_component_10m']
        spd = interp(np.hypot(u, w))
        ui, wi = interp(u), interp(w)
        dirn = (270.0 - np.degrees(np.arctan2(wi, ui))) % 360.0     # meteorological FROM
        wimg = Image.fromarray(colourise(spd, WIND_EDGES, WIND_RGB, transparent_below=False), 'RGBA')
        draw_arrows(wimg, spd, dirn)
        wind.append(webp_b64(wimg))
        span = (h - prev_h) if prev_h is not None else None
        if span:
            rate = interp(np.clip(v['precipitation'], 0, None)) / span
            precip.append(webp_b64(Image.fromarray(colourise(rate, PRECIP_EDGES, PRECIP_RGB), 'RGBA')))
        else:
            precip.append(None)              # first step: no preceding interval
        times.append(vt.strftime('%Y-%m-%dT%H:%MZ'))
        frames.append(h)
        prev_h = h
    n = len(times)
    return {
        'source': 'ecmwf_ifs_9km_elements', 'region': region,
        'run': run.strftime('%y%m%d_%H%M'),
        'generated': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'north': north, 'south': south, 'west': west, 'east': east,
        'img_w': W, 'img_h': H, 'fmt': 'webp', 'mercator_rows': True,
        'frames': frames, 'times': times, 'wind': wind, 'precip': precip,
        'precip_unit': 'mm/h (interval mean)',
        'wind_legends': [legend_b64(WIND_EDGES, WIND_RGB, lambda x: '%d' % x)] + [None] * (n - 1),
        'precip_legends': [legend_b64(PRECIP_EDGES, PRECIP_RGB, lambda x: '%g' % x)] + [None] * (n - 1),
        'credit': CREDIT,
    }


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--region', required=True, choices=sorted(REGIONS))
    ap.add_argument('--run', help='YYYY-MM-DDTHH (default: newest complete 00Z/12Z)')
    ap.add_argument('--out', default='ecmwf_latest.json')
    ap.add_argument('--elements-out', default='ecmwf_elements_latest.json')
    ap.add_argument('--skip-if-run', help="exit 3 without writing if the chosen run == this id ('YYMMDD_HHMM')")
    a = ap.parse_args(argv)
    run, prefix, files = pick_run(a.run)
    if a.skip_if_run and run.strftime('%y%m%d_%H%M') == a.skip_if_run:
        print('run %s already published; nothing to do' % a.skip_if_run)
        return 3
    grid, elements = build(a.region, run, prefix, files)
    import os
    for path, obj in ((a.out, grid), (a.elements_out, elements)):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(obj, f, separators=(',', ':'))
        import gzip
        os.replace(tmp, path)
        raw = open(path, 'rb').read()
        print('wrote %s: %.1f MB (gzip %.1f MB)' % (path, len(raw) / 1e6, len(gzip.compress(raw, 6)) / 1e6))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
