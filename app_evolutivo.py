import os, json
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from ranking import run_daily_rank
from db import maybe_init_db, latest_run, save_run

app = FastAPI(title="Evolutivo Ranking Service", version="0.1.0")

@app.get("/healthz")
def health():
    return {"ok": True}

@app.get("/")
def root():
    return {"message": "Evolutivo ranking listo. Usa /rank/run para ejecutar, /rank/top50 y /rank/top3 para ver resultados."}

@app.post("/rank/run")
def rank_run():
    payload = run_daily_rank()
    conn = maybe_init_db()
    if conn:
        save_run(conn, payload)
    return JSONResponse(payload)

@app.get("/rank/top50")
def rank_top50():
    conn = maybe_init_db()
    if conn:
        row = latest_run(conn)
        if row:
            return JSONResponse({"as_of": row["as_of"].isoformat(), "top50": row["top50"]})
    try:
        with open("/mnt/data/last_rank.json", "r") as f:
            j = json.load(f)
            return JSONResponse({"as_of": j["as_of"], "top50": j["top50"]})
    except Exception:
        return JSONResponse({"error": "No hay resultados aún. Ejecuta POST /rank/run."}, status_code=404)

@app.get("/rank/top3")
def rank_top3():
    conn = maybe_init_db()
    if conn:
        row = latest_run(conn)
        if row:
            return JSONResponse({"as_of": row["as_of"].isoformat(), "top3": row["top3"]})
    try:
        with open("/mnt/data/last_rank.json", "r") as f:
            j = json.load(f)
            return JSONResponse({"as_of": j["as_of"], "top3": j["top3"]})
    except Exception:
        return JSONResponse({"error": "No hay resultados aún. Ejecuta POST /rank/run."}, status_code=404)
