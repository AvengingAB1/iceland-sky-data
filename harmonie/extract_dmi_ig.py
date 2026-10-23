#!/usr/bin/env python3
"""
extract_dmi_ig.py - DMI HARMONIE IG (Iceland-Greenland, 2 km, CC BY 4.0) numeric cloud layers +
visibility -> dmi_latest.json in the same grid schema as harmonie_latest.json / MEPS
(row 0 = north, UTC times, run 'YYMMDD_HHMM', `div` 20 = 5 % steps, `vis` base64 bytes).

This is the SAME model run that vedur.is draws as images (verified: tests/iceland_dmi_vs_vedur.py),
read as numbers instead of 4 posterised colour bands. Source: DMI open data on AWS S3, one GRIB
file per forecast hour (whole IG domain, ~420 MB); only the 4 fields we need are fetched with HTTP
Range requests via the published wgrib2-style .index (~19 MB per hour).

Exit codes: 0 written, 3 newest complete run already published (--skip-if-run), 4 source unreachable.

Usage: python extract_dmi_ig.py [--run YYYY-MM-DDTHH] [--out dmi_latest.json] [--skip-if-run YYMMDD_HHMM]
"""
from __future__ import annotations
import argparse
import base64
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import eccodes
from scipy.spatial import cKDTree

S3 = 'https://dmi-opendata.s3.eu-north-1.amazonaws.com/forecastdata/HARMONIE_IG_SF/HARMONIE_IG_SF_%s_%s.grib'
FIELDS = {'LCDC': 'low', 'MCDC': 'mid', 'HCDC': 'high', 'VIS': 'vis'}
LEVEL = '0 m above ground'
HOURS = 72
# Output grid: vedur's Iceland box plus extra sea to the south/west (up-sun at sunset), ~2.8 km.
BOX = dict(lat0=62.0, lat1=67.2, lon0=-26.0, lon1=-12.0)
DLAT, DLON = 0.025, 0.06
DIV, NODATA = 20, 255
VIS_STEP, VIS_CLEAR = 50, 101
UA = {'User-Agent': 'sunsetglow-data/1.0 github.com/avengingab1'}


class SourceUnreachable(Exception):
    pass


def fname(t):
    return t.strftime('%Y-%m-%dT%H%M%SZ')


def url(run, valid):
    return S3 % (fname(run), fname(valid))


def http(u, rng=None, tries=4, timeout=120):
    h = dict(UA)
    if rng:
        h['Range'] = 'bytes=%d-%d' % rng
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers=h), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                raise
            err = e
        except Exception as e:                               # timeouts, resets
            err = e
        time.sleep(3 * (k + 1))
    raise err


def exists(u):
    try:
        http(u, rng=(0, 15), tries=2, timeout=30)
        return True
    except urllib.error.HTTPError:
        return False


def newest_complete_run(now=None):
    """Newest 3-hourly run whose LAST file (+72 h) and its index are published."""
    now = now or dt.datetime.now(dt.timezone.utc)
    reached = False
    for back in range(0, 8):
        t = now - dt.timedelta(hours=3 * back)
        run = t.replace(hour=t.hour - t.hour % 3, minute=0, second=0, microsecond=0)
        last = run + dt.timedelta(hours=HOURS)
        try:
            ok = exists(url(run, last) + '.index') and exists(url(run, last))
            reached = True
        except Exception:
            continue
        if ok:
            return run
    if not reached:
        raise SourceUnreachable('dmi-opendata S3 not answering')
    return None


def fetch_hour(run, h, latlon=False):
    """{short: 2D field} for the 4 wanted messages of one forecast hour (+ lat/lon if asked)."""
    u = url(run, run + dt.timedelta(hours=h))
    rows = [l.split(':') for l in http(u + '.index').decode().strip().splitlines()]
    offs = [int(r[1]) for r in rows]
    out = {}
    for i, r in enumerate(rows):
        if r[3] in FIELDS and r[4] == LEVEL and r[3] not in out:
            end = offs[i + 1] - 1 if i + 1 < len(offs) else None
            if end is None:                                   # last message: ask the size
                req = urllib.request.Request(u, method='HEAD', headers=UA)
                end = int(urllib.request.urlopen(req, timeout=60).headers['Content-Length']) - 1
            msg = http(u, rng=(offs[i], end))
            g = eccodes.codes_new_from_message(msg)
            try:
                ny, nx = eccodes.codes_get(g, 'Ny'), eccodes.codes_get(g, 'Nx')
                vals = eccodes.codes_get_values(g).astype('float32')
                if eccodes.codes_get(g, 'bitmapPresent'):
                    vals[vals == eccodes.codes_get(g, 'missingValue')] = np.nan
                out[r[3]] = vals
                if latlon and 'lat' not in out:
                    out['lat'] = eccodes.codes_get_array(g, 'latitudes')
                    out['lon'] = eccodes.codes_get_array(g, 'longitudes')
            finally:
                eccodes.codes_release(g)
    missing = [f for f in FIELDS if f not in out]
    if missing:
        raise RuntimeError('hour +%d: fields %s not in %s' % (h, missing, u))
    return out


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode('ascii')


def unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def build(run):
    t0 = time.time()
    first = fetch_hour(run, 0, latlon=True)
    lat, lon = first['lat'], np.where(first['lon'] > 180, first['lon'] - 360, first['lon'])
    near = (lat > BOX['lat0'] - 0.5) & (lat < BOX['lat1'] + 0.5) & (lon > BOX['lon0'] - 1.5) & (lon < BOX['lon1'] + 1.5)
    src = np.flatnonzero(near)
    out_lats = np.round(np.arange(BOX['lat1'], BOX['lat0'] - 1e-9, -DLAT), 5)   # N -> S
    out_lons = np.round(np.arange(BOX['lon0'], BOX['lon1'] + 1e-9, DLON), 5)    # W -> E
    glat, glon = np.meshgrid(out_lats, out_lons, indexing='ij')
    dist, k = cKDTree(unit(lat[src], lon[src])).query(unit(glat, glon).reshape(-1, 3))
    far = dist * 6371.0 > 2.5                                  # > 2.5 km from any 2 km model cell
    pick = src[k]
    shape = (len(out_lats), len(out_lons))

    def sample(fields):                                        # whole domain -> our box, at once
        res = {}
        for f in FIELDS:
            v = fields[f][pick]
            v[far] = np.nan
            res[f] = v.reshape(shape)
        return res

    got = {0: sample(first)}
    del first, lat, lon
    with ThreadPoolExecutor(8) as ex:
        for h, s in zip(range(1, HOURS + 1), ex.map(lambda h: sample(fetch_hour(run, h)), range(1, HOURS + 1))):
            got[h] = s
    print('  fetched %d hours x %d fields in %.0fs' % (len(got), len(FIELDS), time.time() - t0))

    def grid_of(h, f):
        return got[h][f]

    def q_cloud(x):
        return np.where(np.isnan(x), NODATA, np.clip(np.rint(x / 100.0 * DIV), 0, DIV)).astype(np.uint8)

    def q_vis(x):
        return np.where(np.isnan(x), NODATA, np.minimum(np.rint(x / VIS_STEP), VIS_CLEAR)).astype(np.uint8)

    hours = sorted(got)
    times = [(run + dt.timedelta(hours=h)).strftime('%Y-%m-%dT%H:%MZ') for h in hours]
    return {
        'source': 'dmi_harmonie_ig_2km', 'region': 'iceland',
        'scale': 'fraction_div', 'div': DIV, 'nodata': NODATA,
        'run': run.strftime('%y%m%d_%H%M'),
        'generated': dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'lat0': float(out_lats[0]), 'lat1': float(out_lats[-1]),
        'lon0': float(out_lons[0]), 'lon1': float(out_lons[-1]),
        'nlat': len(out_lats), 'nlon': len(out_lons), 'dlat': DLAT, 'dlon': DLON,
        'frames': hours, 'times': times,
        # one base64 string of nlat*nlon bytes per frame (row 0 = north): ~1 byte/cell on phones
        'cloud_enc': 'u8-b64',
        'low': [b64(q_cloud(grid_of(h, 'LCDC'))) for h in hours],
        'mid': [b64(q_cloud(grid_of(h, 'MCDC'))) for h in hours],
        'high': [b64(q_cloud(grid_of(h, 'HCDC'))) for h in hours],
        'vis': [b64(q_vis(grid_of(h, 'VIS'))) for h in hours],
        'vis_enc': 'u8-b64', 'vis_step': VIS_STEP, 'vis_clear': VIS_CLEAR,
        'credit': 'DMI HARMONIE IG (Danish Meteorological Institute open data, CC BY 4.0)',
    }


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', help='YYYY-MM-DDTHH (default: newest complete)')
    ap.add_argument('--out', default='dmi_latest.json')
    ap.add_argument('--skip-if-run')
    a = ap.parse_args(argv)
    try:
        run = (dt.datetime.strptime(a.run, '%Y-%m-%dT%H').replace(tzinfo=dt.timezone.utc)
               if a.run else newest_complete_run())
    except SourceUnreachable as e:
        print('DMI unreachable (%s); keeping the published run' % e)
        return 4
    if run is None:
        raise SystemExit('no complete DMI IG run in the last 24 h')
    rid = run.strftime('%y%m%d_%H%M')
    if a.skip_if_run and rid == a.skip_if_run:
        print('run %s already published; nothing to do' % rid)
        return 3
    print('run=%s' % rid)
    res = build(run)
    tmp = a.out + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(res, f, separators=(',', ':'))
    os.replace(tmp, a.out)
    import gzip
    raw = open(a.out, 'rb').read()
    print('wrote %s: run=%s %dx%d x %d frames, %.1f MB (gzip %.1f MB)' % (
        a.out, rid, res['nlat'], res['nlon'], len(res['times']), len(raw) / 1e6, len(gzip.compress(raw, 6)) / 1e6))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
