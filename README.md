# iceland-sky-data

Public **data feed** for the Sunsetglow app (a personal Iceland sunset‑photography
tool). A scheduled GitHub Action downloads the latest [vedur.is](https://en.vedur.is)
HARMONIE cloud forecast + thattaspa wind/precip maps, extracts them to compact JSON,
and publishes to GitHub Pages:

- `harmonie_latest.json` — low/mid/high cloud okta grids (~3 days, hourly)
- `elements_latest.json` — wind + precipitation map overlays (~7 days)

Served at: `https://avengingab1.github.io/iceland-sky-data/`

This repo contains **only the data‑extraction scripts** (`harmonie/`) and the
workflow. The application itself is maintained separately.

Runs free on a public repo (GitHub Actions are unlimited for public repositories).
