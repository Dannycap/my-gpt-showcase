# Efficient Frontier Explorer

This repository contains:

- **FastAPI backend** providing portfolio optimization endpoints.
- **Next.js + Tailwind frontend** exported statically to the `docs/` folder for GitHub Pages.
- The original **Streamlit app** (`app.py`) kept for reference.

## Backend

The backend exposes an `/efficient-frontier` endpoint that returns summary statistics and
plot data for an efficient frontier computed with [`skfolio`](https://pypi.org/project/skfolio/).

### Run locally

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

## Frontend

The frontend lives in `frontend/` and is built with Next.js and Tailwind.
It calls the FastAPI backend and renders a simple form, table and scatter chart.

### Develop

```bash
cd frontend
npm install
npm run dev
```

### Static export

```bash
cd frontend
npm run build
npm run export  # outputs to ../docs
```

The generated `docs/` folder can be served by GitHub Pages.

## Streamlit reference

`app.py` contains the original Streamlit implementation. It is no longer used
for deployment but remains as a reference implementation.
