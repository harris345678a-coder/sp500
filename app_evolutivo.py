
# -*- coding: utf-8 -*-
"""
app_evolutivo.py — API FastAPI endurecida para exponer el pipeline de ranking.
Endpoints:
- GET /healthz
- GET /rank/run-top3?token=...&audit=0|1
"""
from __future__ import annotations

import os
import time
import logging
from typing import Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse

# -------------------- Logging raíz --------------------

def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name, None)
    if val is None:
        return default
    return str(val).strip().lower() in {"1","true","yes","y","on"}

def _setup_root_logging() -> logging.Logger:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    logger = logging.getLogger("app_evolutivo")

    # Silenciar ruido de 3ros salvo DEBUG
    debug_3p = _env_bool("DEBUG", False)
    third_party = [
        "yfinance","urllib3","peewee",
        "urllib3.connectionpool","uvicorn","asyncio"
    ]
    for name in third_party:
        logging.getLogger(name).setLevel(logging.DEBUG if debug_3p else logging.WARNING)

    return logger

log = _setup_root_logging()

# -------------------- Import robusto --------------------

_import_ok = True
_import_error: Optional[Exception] = None
try:
    from ranking import run_full_pipeline  # type: ignore
except Exception as e:
    _import_ok = False
    _import_error = e
    log.error("No se pudo importar run_full_pipeline desde ranking.py", exc_info=True)

# -------------------- App --------------------

app = FastAPI(title="Ranking Evolutivo", version="1.0.0")

@app.get("/healthz")
def healthz():
    status_obj = {
        "ok": _import_ok,
        "import_error": str(_import_error) if _import_error else None,
        "log_level": os.getenv("LOG_LEVEL","INFO"),
        "debug": _env_bool("DEBUG", False),
    }
    return JSONResponse(status_code=200, content=status_obj)

@app.get("/rank/run-top3")
def run_top3(
    token: str = Query(..., description="Token de acceso"),
    audit: int = Query(default=int(os.getenv("AUDIT_DEFAULT","0")), ge=0, le=1, description="1 para auditoría en payload")
):
    app_token = os.getenv("APP_TOKEN", "123")
    if token != app_token:
        raise HTTPException(status_code=401, detail="token inválido")

    if not _import_ok:
        raise HTTPException(status_code=503, detail=f"ranking.run_full_pipeline no disponible: {_import_error}")

    t0 = time.time()
    try:
        payload = run_full_pipeline(audit=bool(audit))
        took = round(time.time() - t0, 2)
        top50 = payload.get("top50", [])
        top3 = payload.get("top3_factors", [])
        log.info("Pipeline listo · as_of=%s · top50=%d · top3_factors=%d · %.2fs",
                 payload.get("as_of"), len(top50), len(top3), took)
        return JSONResponse(status_code=200, content={
            "ok": True,
            "took_s": took,
            **payload
        })
    except Exception as e:
        took = round(time.time() - t0, 2)
        log.exception("Fallo al ejecutar pipeline (%.2fs)", took)
        raise HTTPException(status_code=500, detail=f"pipeline_error: {type(e).__name__}: {e}")
