
# app_evolutivo.py
# FastAPI app con tolerancia al parámetro `audit`, logging profesional
# y silenciamiento de librerías ruidosas por defecto.
from __future__ import annotations

import inspect
import logging
import os
import time
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

# ------------------------------------------------------------------
# Configuración de logging
# ------------------------------------------------------------------
def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)

DEBUG = _env("DEBUG", "0").lower() in {"1", "true", "yes", "on"}
LOG_LEVEL = _env("LOG_LEVEL", "DEBUG" if DEBUG else "INFO").upper()

_LEVEL = getattr(logging, LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("app_evolutivo")

# Silenciar librerías de terceros salvo que DEBUG=1
noisy_libs = [
    "yfinance",
    "urllib3",
    "peewee",
    "requests",
    "httpx",
    "uvicorn",
    "gunicorn",
    "asyncio",
]
for _lib in noisy_libs:
    logging.getLogger(_lib).setLevel(logging.DEBUG if DEBUG else logging.WARNING)

# ------------------------------------------------------------------
# Carga de ranking.run_full_pipeline con tolerancia a firma
# ------------------------------------------------------------------
try:
    from ranking import run_full_pipeline as _run_full_pipeline  # type: ignore
    RANKING_IMPORTED = True
    log.debug("Importado ranking.run_full_pipeline correctamente.")
except Exception as e:  # pragma: no cover
    RANKING_IMPORTED = False
    _IMPORT_ERROR = e
    _run_full_pipeline = None  # type: ignore
    log.exception("No se pudo importar run_full_pipeline desde ranking.py")

def _supports_audit_kwarg() -> bool:
    if not RANKING_IMPORTED or _run_full_pipeline is None:
        return False
    try:
        sig = inspect.signature(_run_full_pipeline)  # type: ignore[arg-type]
        # Acepta **kwargs
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            return True
        # Acepta 'audit' explícito
        return "audit" in sig.parameters
    except Exception:
        return False

def _safe_run_full_pipeline(audit: bool) -> Dict[str, Any]:
    if not RANKING_IMPORTED or _run_full_pipeline is None:
        raise RuntimeError(f"ranking.run_full_pipeline no disponible: {_IMPORT_ERROR!r}")
    if _supports_audit_kwarg():
        return _run_full_pipeline(audit=audit)  # type: ignore[misc]
    else:
        return _run_full_pipeline()  # type: ignore[misc]

# ------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------
def _boolish(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return default

def _summary(payload: Dict[str, Any]) -> tuple[str | None, int, int]:
    as_of = payload.get("as_of")
    top50 = payload.get("top50") or []
    top3 = payload.get("top3_factors") or payload.get("top3") or []
    return as_of, (len(top50) if isinstance(top50, (list, tuple)) else 0), (
        len(top3) if isinstance(top3, (list, tuple)) else 0
    )

# ------------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------------
app = FastAPI(title="Ranking Evolutivo", version="1.0.0")

@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {
        "ok": True,
        "ranking_imported": RANKING_IMPORTED,
        "supports_audit_kwarg": _supports_audit_kwarg(),
    }

@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "ok": True,
        "msg": "Usa /rank/run-top3?token=...&audit=0|1",
        "endpoints": ["/healthz", "/rank/run-top3"],
    }

@app.get("/rank/run-top3")
def run_top3(
    token: str = Query(..., description="Token de seguridad"),
    audit: str | None = Query(None, description="1/true para activar auditoría de datos"),
) -> JSONResponse:
    expected = _env("APP_TOKEN", "123")
    if not expected or token != expected:
        log.warning("Token inválido en /rank/run-top3")
        raise HTTPException(status_code=401, detail="token inválido")

    audit_default = _boolish(_env("AUDIT_DEFAULT", "0"), False)
    want_audit = _boolish(audit, audit_default)

    t0 = time.perf_counter()
    try:
        payload = _safe_run_full_pipeline(audit=want_audit)
    except TypeError as te:
        # Si a pesar de todo hubiera un TypeError por 'audit', reintenta sin audit.
        log.error("TypeError al invocar run_full_pipeline(audit=?). Reintentando sin 'audit'.")
        log.debug("Detalle TypeError: %r", te)
        payload = _safe_run_full_pipeline(audit=False)
    except Exception as e:  # pragma: no cover
        log.exception("Fallo ejecutando pipeline")
        raise HTTPException(status_code=500, detail=f"pipeline error: {e!r}")

    took = time.perf_counter() - t0
    # Normalizar bandera de éxito
    if "ok" not in payload:
        payload["ok"] = True
    payload["took_s"] = round(float(payload.get("took_s", took)), 2)

    # Log de resumen
    as_of, n50, n3 = _summary(payload)
    log.info("Pipeline listo · as_of=%s · top50=%s · top3_factors=%s · %.2fs", as_of, n50, n3, took)

    return JSONResponse(payload)
