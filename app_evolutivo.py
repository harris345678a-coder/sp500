import os, json, threading, time
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse

from ranking import run_full_pipeline
from db import maybe_init_db, latest_run, save_run
from presend_rules import build_top3_signals  # <- integrado aquí

# === Writable data directory ===
DATA_DIR = os.getenv("DATA_DIR", "/tmp/evolutivo")
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="Evolutivo Signal Service", version="0.3.4")
RUN_TOKEN = os.getenv("RUN_TOKEN", "123")
STATUS_PATH = os.path.join(DATA_DIR, "signal_status.json")
LAST_JSON = os.path.join(DATA_DIR, "last_signals.json")

def _check_token(token: Optional[str]) -> None:
    expected = (RUN_TOKEN or "").strip()
    if expected and token != expected:
        raise HTTPException(status_code=401, detail="Token inválido")

def _write_status(d: dict):
    with open(STATUS_PATH, "w") as f:
        json.dump(d, f)

def _read_status() -> dict:
    try:
        with open(STATUS_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {"state": "idle"}

def _compute_and_finalize_payload() -> dict:
    payload = run_full_pipeline()
    # Integra el pre-send aquí (sin tocar ranking.py)
    top3_factors = payload.get("top3_factors") or payload.get("phase3_top3") or []
    if top3_factors:
        approved_top3, rejected_top3 = build_top3_signals(top3_factors, as_of=payload.get("as_of"))
        payload["approved_top3"] = approved_top3
        payload["rejected_top3"] = rejected_top3
    return payload

def _run_and_persist():
    _write_status({"state": "running", "started_at": time.time()})
    payload = _compute_and_finalize_payload()
    conn = maybe_init_db()
    if conn:
        save_run(conn, payload)
    with open(LAST_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    _write_status({"state": "done", "as_of": payload.get("as_of"), "finished_at": time.time()})

# -------------------- Health/Status --------------------

@app.get("/healthz")
def health():
    return {"ok": True, "data_dir": DATA_DIR, "version": "0.3.4"}

@app.get("/signals/status")
def signals_status():
    return _read_status()

# -------------------- Main run endpoints --------------------

@app.get("/signals/run-top3")
def signals_run_top3(token: Optional[str] = Query(default=None)):
    _check_token(token)
    _run_and_persist()
    try:
        with open(LAST_JSON, "r") as f:
            j = json.load(f)
        return JSONResponse({"as_of": j.get("as_of"), "top3": j.get("approved_top3")})
    except Exception:
        raise HTTPException(status_code=500, detail="No se pudo leer el resultado")

@app.get("/signals/run-redirect")
def signals_run_redirect(token: Optional[str] = Query(default=None)):
    _check_token(token)
    th = threading.Thread(target=_run_and_persist, daemon=True)
    th.start()
    html = f"""
    <html><head><meta charset='utf-8'><title>Ejecutando análisis…</title></head>
    <body style="font-family:Arial; padding:24px;">
      <h3>Ejecutando análisis…</h3>
      <p>Se publicarán las 3 señales aprobadas cuando terminen los 26 filtros.</p>
      <p><b>Data dir:</b> {DATA_DIR}</p>
      <script>
        async function tick(){{
          try{{
            const r = await fetch('/signals/status', {{cache:'no-store'}});
            const j = await r.json();
            if(j.state==='done'){{ window.location.href='/signals/top3'; return; }}
            if(j.state==='error'){{ document.body.innerHTML='<h3>Error</h3><pre>'+JSON.stringify(j,null,2)+'</pre>'; return; }}
          }}catch(e){{}}
          setTimeout(tick, 2000);
        }}
        tick();
      </script>
    </body></html>
    """
    return HTMLResponse(content=html)

@app.get("/signals/top3")
def signals_top3():
    try:
        with open(LAST_JSON, "r") as f:
            j = json.load(f)
        return JSONResponse({"as_of": j.get("as_of"), "top3": j.get("approved_top3")})
    except Exception:
        return JSONResponse({"error": "No hay resultados aún. Ejecuta /rank/run-redirect?token=123"}, status_code=404)

@app.get("/signals/run")
def signals_run(token: Optional[str] = Query(default=None)):
    _check_token(token)
    payload = _compute_and_finalize_payload()
    conn = maybe_init_db()
    if conn:
        save_run(conn, payload)
    with open(LAST_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    return JSONResponse(payload)

# -------------------- Retro-compatibilidad /rank/* --------------------

@app.get("/rank/status")
def old_rank_status():
    return signals_status()

@app.get("/rank/top3")
def old_rank_top3():
    return signals_top3()

@app.get("/rank/run-redirect")
def old_rank_run_redirect(token: Optional[str] = Query(default=None)):
    return signals_run_redirect(token)

@app.get("/rank/run-top3")
@app.get("/rank/run_top3")
def old_rank_run_top3(token: Optional[str] = Query(default=None)):
    return signals_run_top3(token)

@app.get("/rank/run")
def old_rank_run(token: Optional[str] = Query(default=None)):
    return signals_run(token)

# -------------------- Root --------------------

@app.get("/")
def root():
    return {
        "message": "Usa /rank/run-top3?token=123 para ejecutar TODO y ver el Top 3 final (Trigger, SL, TP, estrategia).",
        "data_dir": DATA_DIR,
        "version": "0.3.4",
        "endpoints": [
            "/rank/run?token=123",
            "/rank/run-top3?token=123",
            "/rank/run-redirect?token=123",
            "/rank/top3",
            "/rank/status",
            "/signals/run?token=123",
            "/signals/run-top3?token=123",
            "/signals/run-redirect?token=123",
            "/signals/top3",
            "/signals/status"
        ]
    }
