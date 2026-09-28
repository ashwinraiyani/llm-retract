# Prototype Dashboard (Demo)

This `dashboard/` directory contains a **static, self-contained prototype** for the proposed project:

**AI-Based Heatwave Early Warning and Health Risk Prediction System**

It is intended for project committee demonstration to show what the planned decision-support dashboard and mobile alert framework could look like.

## What this prototype is

- A visual and interactive mockup of the proposed dashboard deliverables.
- Built with **HTML, CSS, and JavaScript** only (no backend).
- Uses **Leaflet** (map) and **Chart.js** (charts) through CDN links.
- Uses district-level **sample/synthetic data only** for Gujarat districts.

## What this prototype is not

- Not a production application.
- Not connected to live IMD/health/GIS government systems.
- Not using real patient or live operational data.

> All displayed values are explicitly labeled as:
> **"SAMPLE / SYNTHETIC DATA FOR DEMONSTRATION PURPOSES ONLY — NOT ACTUAL GOVERNMENT DATA"**

## Run locally

### Option 1: Open directly
Open `/dashboard/index.html` in a web browser.

### Option 2: Serve from a simple static server (recommended)

```bash
cd dashboard
python -m http.server 8000
```

Then open: `http://localhost:8000`

## Files

- `index.html` — dashboard structure and sections
- `styles.css` — visual styling
- `app.js` — map + chart + interaction logic
- `data/districts.js` — synthetic district data used by the prototype

## Proposal-to-feature mapping

| Prototype feature | Proposal section alignment | How this prototype demonstrates it |
|---|---|---|
| HHRI components (Exposure, Sensitivity, Adaptive Capacity) and composite risk score | **Section 8** (Conceptual Framework / HHRI) | District panel shows component breakdown and derived HHRI score |
| Decision-support dashboard with map, current conditions, forecast, and risk indicators | **Section 14** (Dashboard design) | Interactive map and district detail panel with current + forecast + health-risk metrics |
| Alert legend (Green → Yellow → Orange → Red) with operational meaning | **Section 15** (Alert framework) | Visible alert tier legend and district-level alert color coding |
| Mobile alert preview | **Section 15** (Mobile Alert Framework) | Mock push/SMS preview for a high-risk district |

## Notes

- Districts and values are synthetic and selected only to represent different heat-risk environments for committee demonstration.
- The prototype is intentionally lightweight so it can run quickly during review meetings.
