# Static Dashboard (GitHub-friendly)

This report is a **non-interactive fallback** for GitHub viewing.
For interactive charts, use `reports/dashboard.html` via a local server.

## Snapshot

| Metric | Value |
|---|---:|
| Total records | 20,000 |
| Final smoothed health | 55.00% |
| Final status | WARNING |
| Layer 1 anomalies | 14,550 (72.75%) |
| Layer 2 anomalies | 12,513 (62.56%) |
| Latest RUL estimate | STABLE |
| Active RUL countdown points | 7,618 |
| Min smoothed health | 55.00% |
| Max smoothed health | 100.00% |

## Health status distribution

| Status | Count |
|---|---:|
| GOOD | 8,085 |
| WARNING | 11,915 |
| DEGRADED | 0 |
| CRITICAL | 0 |

## Recent rows (tail)

### Health index (last 10)

| Timestamp | Raw health | Smoothed health | Status |
|---|---:|---:|---|
| 2025-01-01 05:33:10 | 55.0 | 55.0 | WARNING |
| 2025-01-01 05:33:11 | 55.0 | 55.0 | WARNING |
| 2025-01-01 05:33:12 | 55.0 | 55.0 | WARNING |
| 2025-01-01 05:33:13 | 55.0 | 55.0 | WARNING |
| 2025-01-01 05:33:14 | 55.0 | 55.0 | WARNING |
| 2025-01-01 05:33:15 | 55.0 | 55.0 | WARNING |
| 2025-01-01 05:33:16 | 55.0 | 55.0 | WARNING |
| 2025-01-01 05:33:17 | 55.0 | 55.0 | WARNING |
| 2025-01-01 05:33:18 | 55.0 | 55.0 | WARNING |
| 2025-01-01 05:33:19 | 55.0 | 55.0 | WARNING |


### RUL (last 10)

| Timestamp | Smoothed health | Status | Estimated RUL |
|---|---:|---|---|
| 2025-01-01 05:33:10 | 55.0 | WARNING | STABLE |
| 2025-01-01 05:33:11 | 55.0 | WARNING | STABLE |
| 2025-01-01 05:33:12 | 55.0 | WARNING | STABLE |
| 2025-01-01 05:33:13 | 55.0 | WARNING | STABLE |
| 2025-01-01 05:33:14 | 55.0 | WARNING | STABLE |
| 2025-01-01 05:33:15 | 55.0 | WARNING | STABLE |
| 2025-01-01 05:33:16 | 55.0 | WARNING | STABLE |
| 2025-01-01 05:33:17 | 55.0 | WARNING | STABLE |
| 2025-01-01 05:33:18 | 55.0 | WARNING | STABLE |
| 2025-01-01 05:33:19 | 55.0 | WARNING | STABLE |


## Open interactive dashboard locally

```bash
python3 -m http.server 8000
# then open http://localhost:8000/reports/dashboard.html
```
