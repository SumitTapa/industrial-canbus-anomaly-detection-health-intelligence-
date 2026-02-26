# Reports Usage

This folder contains generated report artifacts.

## Why `industrial_eda_report.html` may look "broken" on GitHub

If you open the file with a GitHub `.../blob/...` URL, GitHub displays the source code and does **not** run JavaScript.
The EDA and dashboard files are interactive Plotly HTML pages, so charts render only when opened directly or served from a web server.

## How to view properly

```bash
python3 -m http.server 8000
```

Then open:

- `http://localhost:8000/reports/industrial_eda_report.html`
- `http://localhost:8000/reports/dashboard.html`

## Artifacts in this directory

- `industrial_eda_report.html` - interactive EDA report
- `dashboard.html` - interactive dashboard
- `dashboard_static.md` - GitHub-friendly static dashboard snapshot
- `anomaly_detection_report.pdf` - static PDF report (generated locally; not committed because binary files are unsupported)
- `model_comparison_insights.md` - narrative model comparison notes


To regenerate the static dashboard snapshot:

```bash
python3 reports/generate_static_report.py
```


## Host from branch with GitHub Pages

A workflow is included at `.github/workflows/deploy-reports-pages.yml` that deploys this `reports/` folder to GitHub Pages on push.

After enabling Pages (GitHub Actions mode), use:

- `https://<owner>.github.io/<repo>/dashboard.html`
- `https://<owner>.github.io/<repo>/industrial_eda_report.html`
- `https://<owner>.github.io/<repo>/dashboard_static.md`


To regenerate the PDF report locally:

```bash
python3 reports/generate_pdf_report.py
```
