#!/usr/bin/env python3
"""carry_forward.py - keep the recent PAST hours in the published cloud grid.

GitHub Pages only ever serves the newest run. A new run starts at its issue time (+1 h for
vedur HARMONIE, +0 h for MEPS), so as soon as it is published the hours just before it
(this evening's sunset, say) can no longer be downloaded, and a device that didn't fetch the
intermediate runs falls back to a much older cached run for them.

This step prepends to the NEW grid the frames of the CURRENTLY PUBLISHED grid whose valid
time is before the new run's first frame and within --hours of it. The published file
already holds the newest data for those hours (it was built the same way), so every hour in
the output comes from the newest run that ever covered it. Re-running on the same run is
idempotent. Frames are only carried when the grid geometry matches exactly; otherwise the
new file is left untouched.

Adds `frame_runs` (source run id per frame) so the app can tell carried hours apart.

Usage: python carry_forward.py --new _site/harmonie_latest.json --prev-url <published url> [--hours 30]
"""
import argparse
import datetime as dt
import json
import os
import sys
import urllib.request

GEOM = ('lat0', 'lat1', 'lon0', 'lon1', 'nlat', 'nlon', 'nodata')
LAYERS = ('low', 'mid', 'high')
OPTIONAL = ('vis',)          # per-frame optional layers (MEPS visibility); null frame = not available


def parse_t(s):
    return dt.datetime.strptime(s, '%Y-%m-%dT%H:%MZ').replace(tzinfo=dt.timezone.utc)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--new', required=True)
    ap.add_argument('--prev-url')
    ap.add_argument('--prev-file')
    ap.add_argument('--hours', type=float, default=30)
    a = ap.parse_args(argv)
    new = json.load(open(a.new, encoding='utf-8'))
    run = new.get('run', '')
    new.setdefault('frame_runs', [run] * len(new['times']))
    try:
        if a.prev_file:
            prev = json.load(open(a.prev_file, encoding='utf-8'))
        else:
            req = urllib.request.Request(a.prev_url, headers={'User-Agent': 'sunsetglow-carry/1.0'})
            prev = json.load(urllib.request.urlopen(req, timeout=120))
    except Exception as e:
        print('carry_forward: no previous grid (%s); unchanged' % str(e)[:120])
        prev = None
    carried = 0
    if prev and prev.get('times') and new.get('times'):
        same = all(prev.get(k) == new.get(k) for k in GEOM) and (prev.get('div') or 8) == (new.get('div') or 8)
        if not same:
            print('carry_forward: grid geometry/scale differs from published file; unchanged')
        else:
            t0 = parse_t(new['times'][0])
            lo = t0 - dt.timedelta(hours=a.hours)
            pruns = prev.get('frame_runs') or [prev.get('run', '')] * len(prev['times'])
            idx = [i for i, s in enumerate(prev['times']) if lo <= parse_t(s) < t0]
            if idx:
                carried = len(idx)
                pf = prev.get('frames') or [None] * len(prev['times'])
                for k in LAYERS:
                    new[k] = [prev[k][i] for i in idx] + new[k]
                for k in OPTIONAL:
                    if k in new:
                        pk = prev.get(k) if isinstance(prev.get(k), list) and len(prev[k]) == len(prev['times']) else None
                        new[k] = [(pk[i] if pk else None) for i in idx] + new[k]
                new['times'] = [prev['times'][i] for i in idx] + new['times']
                new['frame_runs'] = [pruns[i] for i in idx] + new['frame_runs']
                if 'frames' in new:
                    rt = dt.datetime.strptime(run, '%y%m%d_%H%M').replace(tzinfo=dt.timezone.utc)
                    new['frames'] = [int(round((parse_t(prev['times'][i]) - rt).total_seconds() / 3600)) for i in idx] + new['frames']
    new['carried_frames'] = carried
    tmp = a.new + '.tmp'                     # atomic: never leave a half-written feed file
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(new, f, separators=(',', ':'))
    os.replace(tmp, a.new)
    print('carry_forward: run %s, carried %d past frame(s) -> %d frames %s..%s' % (
        run, carried, len(new['times']), new['times'][0], new['times'][-1]))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
