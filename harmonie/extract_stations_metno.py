#!/usr/bin/env python3
"""
extract_stations_metno.py - point forecasts for the North Norway places from MET Norway
Locationforecast 2.0 (api.met.no, free, NLOD/CC BY 4.0; a descriptive User-Agent is
REQUIRED by their terms), written in the SAME schema as Iceland's vedur_stations.json so
the app's W (weather-symbol) layer works unchanged:

  { source, issued, fetched, times:[ISO], interval:[h], labels:{code:text}, overview:[sid],
    stations:{ sid:{name,lat,lon,area, W:[], T:[], N:[], F:[], D2:[], R:[]} } }

MET symbol codes (clearsky_day, lightrainshowers_night, ...) are mapped onto the vedur
weather codes 1..23 that the app's own SVG icons (wicons.js) are keyed on.

Three files (one shared time axis): stations.json = named places + towns (old apps read only this),
stations_peaks.json = peaks at summit altitude with fog G (summit in cloud), stations_grid.json = land grid.

Usage: python extract_stations_metno.py [--out stations.json] [--tiers towns,peaks,grid] [--limit N]
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

API = 'https://api.met.no/weatherapi/locationforecast/2.0/compact'
UA = {'User-Agent': 'sunsetglow-data/1.0 https://github.com/avengingab1/norway-sky-data'}

# sid: (name, lat, lon, area)  -- same coordinates as web/js/places.js NNORWAY.
PLACES = {
    'tromso': ('Tromsø', 69.65163, 18.95586, 'troms'),
    'reine': ('Reine', 67.93247, 13.08877, 'lofoten'), 'hamnoy': ('Hamnøy', 67.94639, 13.13762, 'lofoten'),
    'a_lofoten': ('Å', 67.87962, 12.98404, 'lofoten'), 'kvalvika': ('Kvalvika', 68.07997, 13.08043, 'lofoten'),
    'ramberg': ('Ramberg', 68.08991, 13.22989, 'lofoten'), 'nusfjord': ('Nusfjord', 68.03541, 13.3477, 'lofoten'),
    'haukland': ('Haukland', 68.19799, 13.52919, 'lofoten'), 'eggum': ('Eggum', 68.30812, 13.68472, 'lofoten'),
    'leknes': ('Leknes', 68.14746, 13.61153, 'lofoten'), 'henningsvaer': ('Henningsvær', 68.15289, 14.20072, 'lofoten'),
    'svolvaer': ('Svolvær', 68.23217, 14.5646, 'lofoten'),
    'andenes': ('Andenes', 69.3123, 16.10571, 'vesteralen'), 'sto': ('Stø', 69.01883, 15.12397, 'vesteralen'),
    'segla': ('Segla', 69.50737, 17.59572, 'senja'), 'tungeneset': ('Tungeneset', 69.48695, 17.33304, 'senja'),
    'bergsbotn': ('Bergsbotn', 69.43964, 17.46757, 'senja'), 'husoy': ('Husøy', 69.5442, 17.6637, 'senja'),
    'hamn_senja': ('Hamn i Senja', 69.41512, 17.21293, 'senja'),
    'ersfjordbotn': ('Ersfjordbotn', 69.69441, 18.62083, 'troms'), 'sommaroy': ('Sommarøy', 69.63357, 18.0074, 'troms'),
    'grotfjord': ('Grøtfjord', 69.77701, 18.53977, 'troms'), 'tromvik': ('Tromvik', 69.77939, 18.39909, 'troms'),
    'lyngseidet': ('Lyngseidet', 69.57626, 20.21907, 'lyngen'), 'lakselvbukt': ('Lakselvbukt', 69.43331, 19.64699, 'lyngen'),
    'skibotn': ('Skibotn', 69.3908, 20.26784, 'lyngen'),
    'bodo': ('Bodø', 67.28443, 14.38175, 'salten'), 'kjerringoy': ('Kjerringøy', 67.52078, 14.76455, 'salten'),
    'narvik': ('Narvik', 68.43838, 17.4272, 'ofoten'),
    'alta': ('Alta', 69.9666, 23.27333, 'finnmark'), 'hammerfest': ('Hammerfest', 70.64134, 23.83608, 'finnmark'),
    'hasvik': ('Hasvik', 70.63289, 22.11576, 'finnmark'),
}
OVERVIEW = ['tromso', 'bodo', 'narvik', 'alta', 'svolvaer', 'leknes', 'reine', 'andenes',
            'hammerfest', 'lyngseidet', 'segla', 'hasvik']

LABELS = {1: 'Clear sky', 2: 'Partly cloudy', 3: 'Cloudy', 4: 'Overcast', 5: 'Light rain', 6: 'Rain',
          7: 'Light sleet', 8: 'Sleet', 9: 'Light snow', 10: 'Snow', 11: 'Rain showers', 12: 'Sleet shower',
          13: 'Hail', 14: 'Dust devil', 15: 'Dust storm', 16: 'Blowing snow', 17: 'Fog', 18: 'Light drizzle',
          19: 'Drizzle', 20: 'Freezing rain', 21: 'Hail', 22: 'Light thunder', 23: 'Thunder'}


def w_code(sym):
    """MET symbol_code -> vedur weather code (1..23)."""
    if not sym:
        return 0
    s = sym.split('_')[0]                           # drop _day/_night/_polartwilight
    if 'thunder' in s:
        return 22 if s.startswith('light') else 23
    if s == 'clearsky':
        return 1
    if s == 'fair':
        return 2
    if s == 'partlycloudy':
        return 3
    if s == 'cloudy':
        return 4
    if s == 'fog':
        return 17
    showers = 'showers' in s
    if 'sleet' in s:
        return 12 if showers else (7 if s.startswith('light') else 8)
    if 'snow' in s:
        return 9 if (s.startswith('light') or showers) else 10
    if 'rain' in s:
        if showers:
            return 11
        return 5 if s.startswith('light') else 6
    return 0


API_COMPLETE = 'https://api.met.no/weatherapi/locationforecast/2.0/complete'
WORKERS, PAUSE = 8, 0.4             # each worker waits 0.4 s + request time: < 20 req/s (MET's guidance) even on a fast link
TIER_FILES = {'towns': None, 'peaks': 'stations_peaks.json', 'grid': 'stations_grid.json'}


def fetch(lat, lon, altitude=None, complete=False):
    params = {'lat': '%.4f' % lat, 'lon': '%.4f' % lon}
    if altitude is not None:
        params['altitude'] = int(altitude)
    for k in range(4):
        try:
            r = requests.get(API_COMPLETE if complete else API, params=params, headers=UA, timeout=40)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(10 * (k + 1)); continue
            r.raise_for_status()
        except requests.RequestException:
            time.sleep(10 * (k + 1))
    return None


def load_points(tiers):
    """{tier: {sid: (name, lat, lon, area, alt)}}. 'towns' = the 32 named places + the town fill
    tier; 'peaks' (summit altitude) and 'grid' from station_points.json
    (tools/build_nn_station_points.py). Every point is a MET model point forecast."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'station_points.json')
    extra = json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}
    if isinstance(extra, list):                     # older single-list format = towns
        extra = {'towns': extra}
    out = {}
    if 'towns' in tiers:
        out['towns'] = {sid: (n, la, lo, ar, None) for sid, (n, la, lo, ar) in PLACES.items()}
        for sid, n, la, lo in extra.get('towns', []):
            out['towns'].setdefault(sid, (n, la, lo, '', None))
    if 'peaks' in tiers:
        out['peaks'] = {sid: (n, la, lo, 'peak', alt) for sid, n, la, lo, alt in extra.get('peaks', [])}
    if 'grid' in tiers:
        out['grid'] = {sid: (n, la, lo, 'grid', None) for sid, n, la, lo in extra.get('grid', [])}
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='stations.json', help='towns file; peaks/grid files are written next to it')
    ap.add_argument('--tiers', default='towns,peaks,grid')
    ap.add_argument('--limit', type=int, default=0, help='max points per tier (testing)')
    ap.add_argument('--region', default='nnorway', help="'iceland': peaks + grid only (its towns are vedur stations)")
    a = ap.parse_args(argv)
    tiers = [x for x in a.tiers.split(',') if x in TIER_FILES]
    points = load_points(tiers)
    if a.limit:
        points = {k: dict(list(v.items())[:a.limit]) for k, v in points.items()}
    jobs = [(tier, sid, v) for tier in tiers for sid, v in points.get(tier, {}).items()]

    def work(job):
        tier, sid, (name, lat, lon, area, alt) = job
        time.sleep(PAUSE)
        # Peaks: the FULL forecast at the summit altitude, for fog_area_fraction (= summit in cloud).
        return job, fetch(lat, lon, altitude=alt, complete=(tier == 'peaks'))

    t0 = time.time()
    raw, issued, failed = {}, None, []
    with ThreadPoolExecutor(WORKERS) as ex:
        for (tier, sid, _), j in ex.map(work, jobs):
            if j:
                raw[(tier, sid)] = j
                issued = issued or j['properties']['meta'].get('updated_at')
            else:
                failed.append(sid)
    print('fetched %d/%d points in %.0fs%s' % (len(raw), len(jobs), time.time() - t0,
                                             (' (failed: %s)' % ', '.join(failed[:10])) if failed else ''), flush=True)
    if not raw or ('towns' in tiers and not any(k[0] == 'towns' for k in raw)):
        raise SystemExit('no forecasts fetched')
    # ONE time axis for every tier (hourly first ~60 h, then 6-hourly) so the app can merge files.
    tset = sorted({e['time'] for j in raw.values() for e in j['properties']['timeseries']})
    times = [t.replace(':00:00Z', ':00Z') for t in tset]
    tpos = {t: i for i, t in enumerate(tset)}
    interval = []
    for i in range(len(tset)):
        if i + 1 < len(tset):
            d = (dt.datetime.fromisoformat(tset[i + 1].replace('Z', '+00:00')) -
                 dt.datetime.fromisoformat(tset[i].replace('Z', '+00:00'))).total_seconds() / 3600
            interval.append(int(round(d)))
        else:
            interval.append(interval[-1] if interval else 1)
    n = len(tset)
    fetched = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%MZ')
    for tier in tiers:
        stations = {}
        for (tr, sid), j in raw.items():
            if tr != tier:
                continue
            name, lat, lon, area, alt = points[tier][sid]
            st = {'name': name, 'lat': lat, 'lon': lon, 'area': area,
                  'W': [None] * n, 'T': [None] * n, 'N': [None] * n, 'F': [None] * n, 'D2': [None] * n, 'R': [None] * n}
            if tier == 'peaks':
                st['alt'] = alt
                st['G'] = [None] * n                 # fog fraction AT the summit: >= 0.5 = in cloud
            for e in j['properties']['timeseries']:
                i = tpos[e['time']]
                det = e['data']['instant']['details']
                nxt = e['data'].get('next_1_hours') or e['data'].get('next_6_hours') or {}
                st['W'][i] = w_code((nxt.get('summary') or {}).get('symbol_code'))
                st['T'][i] = round(det['air_temperature']) if 'air_temperature' in det else None
                st['N'][i] = round(det['cloud_area_fraction'] / 100, 2) if 'cloud_area_fraction' in det else None
                st['F'][i] = round(det['wind_speed']) if 'wind_speed' in det else None
                st['D2'][i] = round(det['wind_from_direction']) if 'wind_from_direction' in det else None
                st['R'][i] = (nxt.get('details') or {}).get('precipitation_amount')
                if tier == 'peaks' and 'fog_area_fraction' in det:
                    st['G'][i] = round(det['fog_area_fraction'] / 100, 2)
            stations[sid] = st
        out = {
            'source': 'MET Norway Locationforecast 2.0 (api.met.no), NLOD 2.0 / CC BY 4.0',
            'region': a.region, 'tier': tier,
            'fetched': fetched, 'issued': (issued or '').replace(':00:00Z', ':00Z'),
            'times': times, 'interval': interval,
            'labels': {str(k): v for k, v in LABELS.items()},
            'overview': [s for s in OVERVIEW if s in stations] if tier == 'towns' else [],
            'stations': stations,
        }
        path = a.out if TIER_FILES[tier] is None else os.path.join(os.path.dirname(os.path.abspath(a.out)), TIER_FILES[tier])
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, separators=(',', ':'))
        os.replace(tmp, path)
        print('wrote %s: %d %s points x %d times' % (path, len(stations), tier, n))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
