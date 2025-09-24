# app_evolutivo.py — FastAPI + logging fino y sin ruido
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

# ---- Configuración de logging (silenciando librerías ruidosas) ----
def _str2bool(v: str | None, default=False) -> bool:
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "t", "yes", "y", "on"}

def configure_logging() -> None:
    # Nivel principal
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    # Formato compacto y con hora
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)

    # Quita ruido de terceros si no estamos en DEBUG explícito
    debug_all = _str2bool(os.getenv("DEBUG"), False) or level <= logging.DEBUG
    third_party_level = logging.DEBUG if debug_all else logging.WARNING
    for noisy in ["yfinance", "urllib3", "peewee", "asyncio", "PIL", "matplotlib"]:
        logging.getLogger(noisy).setLevel(third_party_level)

    # Asegura que nuestro logger principal esté al nivel elegido
    logging.getLogger("app_evolutivo").setLevel(level)
    logging.getLogger("ranking").setLevel(level)

configure_logging()
log = logging.getLogger("app_evolutivo")

# ---- App ----
app = FastAPI(title="Evolutivo API", version="1.0.0")

# Import retardado para que el logging esté configurado antes
try:
    from ranking import run_full_pipeline
except Exception as ex:
    log.exception("No se pudo importar run_full_pipeline desde ranking.py")
    raise

@app.get("/rank/run-top3")
def run_top3(
    token: str = Query(..., description="Token de ejecución simple"),
    audit: bool = Query(False, description="True para imprimir DATA_AUDIT"),
):
    required = os.getenv("RUN_TOKEN")
    if required and token != required:
        log.warning("Token inválido recibido")
        return JSONResponse({"ok": False, "error": "invalid token"}, status_code=403)

    t0 = datetime.now(timezone.utc)
    payload = run_full_pipeline(audit=audit or _str2bool(os.getenv("AUDIT"), False))

    # Resumen visible siempre (sin ruido)
    as_of = payload.get("as_of")
    top50 = payload.get("top50", [])
    top3 = payload.get("top3_factors", [])
    log.info(
        "Pipeline listo · as_of=%s · top50=%s · top3_factors=%s",
        as_of, len(top50), len(top3)
    )

    # Diagnóstico si algo quedó vacío
    if not top3:
        cause = "top3 vacío"
        if not top50:
            cause = "top50 vacío"
        log.debug("Diagnóstico: %s", cause)

    dt = (datetime.now(timezone.utc) - t0).total_seconds()
    payload_out = {"ok": True, "took_s": round(dt, 2), **payload}
    return JSONResponse(payload_out)