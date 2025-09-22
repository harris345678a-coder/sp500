import os
import json
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse

from ranking import run_daily_rank
from db import maybe_init_db, latest_run, save_run

app = FastAPI(title="Evolutivo Ranking Service", version="0.1.1")

# Token de seguridad para force-run por URL (GET).
# Puedes definir RUN_TOKEN en Render (Environment). Si no está definido, por defecto es "123".
RUN_TOKEN = os.getenv("RUN_TOKEN", "123")


def _check_token(token: Optional[str]) -> None:
    """
    Verifica el token si RUN_TOKEN está definido (no vacío).
    Lanza 401 si no coincide.
    """
    expected = (RUN_TOKEN or "").strip()
    if expected and token != expected:
        raise HTTPException(status_code=401, detail="Token inválido")


def _run_and_save() -> dict:
    """
    Ejecuta el análisis completo (Top 50 y Top 3), persiste resultados en Postgres si hay
    DATABASE_URL, y además guarda un respaldo local en /mnt/data/last_rank.json.
    Devuelve el payload completo.
    """
    payload = run_daily_rank()
    conn = maybe_init_db()
    if conn:
        save_run(conn, payload)
    # Respaldo local para entornos sin DB
    try:
        os.makedirs("/mnt/data", exist_ok=True)
        with open("/mnt/data/last_rank.json", "w") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        # No interrumpimos la respuesta si el respaldo local falla
        pass
    return payload


@app.get("/healthz")
def health():
    return {"ok": True}


@app.get("/")
def root():
    return {
        "message": "Evolutivo listo. Ejecuta /rank/run (GET o POST) para correr el análisis; usa /rank/top50 y /rank/top3 para ver resultados.",
        "force_run_examples": [
            "/rank/run?token=123",
            "/rank/run-top3?token=123",
            "/rank/run-redirect?token=123",
        ],
    }


# ---------- Force run por URL (GET) ----------

@app.get("/rank/run")
def rank_run_get(token: Optional[str] = Query(default=None)):
    """
    Ejecuta el análisis completo y devuelve Top50 + Top3.
    Protegido por token si RUN_TOKEN está definido.
    """
    _check_token(token)
    payload = _run_and_save()
    return JSONResponse(payload)


@app.get("/rank/run-top3")
def rank_run_top3(token: Optional[str] = Query(default=None)):
    """
    Ejecuta el análisis completo y devuelve solo el Top 3.
    Protegido por token si RUN_TOKEN está definido.
    """
    _check_token(token)
    payload = _run_and_save()
    return JSONResponse({"as_of": payload["as_of"], "top3": payload["top3"]})


@app.get("/rank/run-redirect")
def rank_run_redirect(token: Optional[str] = Query(default=None)):
    """
    Ejecuta el análisis completo y redirige a /rank/top3 (303).
    Protegido por token si RUN_TOKEN está definido.
    """
    _check_token(token)
    _run_and_save()
    return RedirectResponse(url="/rank/top3", status_code=303)


# ---------- POST clásico (se mantiene para clientes programáticos) ----------

@app.post("/rank/run")
def rank_run():
    """
    Ejecuta el análisis completo y devuelve Top50 + Top3.
    """
    payload = _run_and_save()
    return JSONResponse(payload)


# ---------- Lecturas ----------

@app.get("/rank/top50")
def rank_top50():
    conn = maybe_init_db()
    if conn:
        row = latest_run(conn)
        if row:
            return JSONResponse({"as_of": row["as_of"].isoformat(), "top50": row["top50"]})
    # Fallback local
    try:
        with open("/mnt/data/last_rank.json", "r") as f:
            j = json.load(f)
            return JSONResponse({"as_of": j["as_of"], "top50": j["top50"]})
    except Exception:
        return JSONResponse({"error": "No hay resultados aún. Ejecuta /rank/run."}, status_code=404)


@app.get("/rank/top3")
def rank_top3():
    conn = maybe_init_db()
    if conn:
        row = latest_run(conn)
        if row:
            return JSONResponse({"as_of": row["as_of"].isoformat(), "top3": row["top3"]})
    # Fallback local
    try:
        with open("/mnt/data/last_rank.json", "r") as f:
            j = json.load(f)
            return JSONResponse({"as_of": j["as_of"], "top3": j["top3"]})
    except Exception:
        return JSONResponse({"error": "No hay resultados aún. Ejecuta /rank/run."}, status_code=404)
