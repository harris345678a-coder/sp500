
# app_evolutivo.py
# FastAPI con envío robusto a Telegram del Top 3 (sin saltarse filtros),
# tolerancia a la firma de ranking.run_full_pipeline, deduplicación,
# y logging profesional.
from __future__ import annotations

import inspect
import logging
import os
import time
import json
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Tuple

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

# ------------------------------------------------------------------
# Helpers de entorno
# ------------------------------------------------------------------
def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)

DEBUG = _env("DEBUG", "0").lower() in {"1", "true", "yes", "on"}
LOG_LEVEL = _env("LOG_LEVEL", "DEBUG" if DEBUG else "INFO").upper()

# ------------------------------------------------------------------
# Configuración de logging
# ------------------------------------------------------------------
_LEVEL = getattr(logging, LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("app_evolutivo")

# Silenciar librerías ruidosas salvo que DEBUG=1
for _lib in ["yfinance", "urllib3", "peewee", "requests", "httpx", "uvicorn", "gunicorn", "asyncio"]:
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
# Dedupe + envío a Telegram
# ------------------------------------------------------------------
def _extract_top3(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Solo señales que YA pasaron por todos los filtros del pipeline.
    top3 = payload.get("top3_factors") or payload.get("top3") or []
    # Normalizar a lista de dicts con 'ticker' y 'reasons'
    norm = []
    for x in top3:
        if isinstance(x, dict) and "ticker" in x:
            reasons = x.get("reasons")
            if isinstance(reasons, (list, tuple)):
                reasons = list(reasons)
            elif reasons is None:
                reasons = []
            else:
                reasons = [str(reasons)]
            norm.append({"ticker": str(x["ticker"]), "reasons": reasons})
    return norm

def _signature(top3: List[Dict[str, Any]], as_of: str | None) -> str:
    tickers = ",".join([t["ticker"] for t in top3])
    base = f"{as_of or ''}|{tickers}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def _state_path() -> str:
    return _env("TOP3_STATE_PATH", "/tmp/top3_last.json")

def _load_state() -> Dict[str, Any]:
    path = _state_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_state(state: Dict[str, Any]) -> None:
    path = _state_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
    except Exception as e:  # pragma: no cover
        log.warning("No pude guardar estado de dedupe en %s: %r", path, e)

def _should_send(sig: str, min_minutes: int) -> Tuple[bool, str]:
    state = _load_state()
    last_hash = state.get("last_hash")
    last_sent_at = state.get("last_sent_at")
    if last_hash and last_sent_at and last_hash == sig:
        try:
            last_dt = datetime.fromisoformat(last_sent_at)
        except Exception:
            last_dt = datetime.now(timezone.utc) - timedelta(days=1)
        elapsed = datetime.now(timezone.utc) - last_dt
        if elapsed < timedelta(minutes=min_minutes):
            return False, f"duplicate_recent({int(elapsed.total_seconds())}s ago)"
    return True, "ok_to_send"

def _save_sent(sig: str, top3: List[Dict[str, Any]], as_of: str | None) -> None:
    _save_state({
        "last_hash": sig,
        "last_sent_at": datetime.now(timezone.utc).isoformat(),
        "as_of": as_of,
        "tickers": [t["ticker"] for t in top3],
    })

def _build_message(as_of: str | None, top3: List[Dict[str, Any]], top50_count: int | None) -> str:
    header = f"📈 Top 3 señales (as_of={as_of})"
    lines = [header, ""]
    for i, item in enumerate(top3, 1):
        reasons = ", ".join(item.get("reasons") or [])
        lines.append(f"{i}) *{item['ticker']}* — {reasons}")
    if top50_count is not None:
        lines.append("")
        lines.append(f"Universo filtrado: {top50_count} candidatos")
    return "\n".join(lines)

def _telegram_send(text: str) -> Tuple[bool, str]:
    token = _env("TELEGRAM_BOT_TOKEN")
    chat_id = _env("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return False, "missing_env(TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID)"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    backoffs = [0.5, 1.0, 2.0]
    last_err = None
    for b in backoffs:
        try:
            r = requests.post(url, json=payload, timeout=10)
            if r.ok and r.json().get("ok"):
                return True, "sent"
            last_err = f"status={r.status_code} body={r.text[:200]}"
        except Exception as e:
            last_err = repr(e)
        time.sleep(b)
    return False, f"send_failed({last_err})"

# ------------------------------------------------------------------
# FastAPI
# ------------------------------------------------------------------
app = FastAPI(title="Ranking Evolutivo", version="1.1.0")

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
        # Si aun así hubiera un TypeError por 'audit', reintenta sin audit.
        log.error("TypeError al invocar run_full_pipeline(audit=?). Reintentando sin 'audit'.")
        log.debug("Detalle TypeError: %r", te)
        payload = _safe_run_full_pipeline(audit=False)
    except Exception as e:  # pragma: no cover
        log.exception("Fallo ejecutando pipeline")
        raise HTTPException(status_code=500, detail=f"pipeline error: {e!r}")

    took = time.perf_counter() - t0
    if "ok" not in payload:
        payload["ok"] = True
    payload["took_s"] = round(float(payload.get("took_s", took)), 2)

    # Resumen
    as_of, n50, n3 = _summary(payload)
    log.info("Pipeline listo · as_of=%s · top50=%s · top3_factors=%s · %.2fs", as_of, n50, n3, took)

    # --------------------------------------------------------------
    # Envío a Telegram: SOLO si hay señales que YA pasaron filtros.
    # Sin duplicados y con ventana mínima de reenvío.
    # --------------------------------------------------------------
    top3 = _extract_top3(payload)
    tele_enabled = _env("TELEGRAM_ENABLED", "1").lower() in {"1", "true", "yes", "on"}
    min_minutes = int(_env("TOP3_MIN_RENOTIFY_MINUTES", "360"))  # 6h por defecto

    notify_info: Dict[str, Any] = {
        "attempted": False,
        "enabled": tele_enabled,
        "reason": None,
        "tickers": [x["ticker"] for x in top3],
    }

    if tele_enabled and top3:
        sig = _signature(top3, as_of)
        ok_to_send, why = _should_send(sig, min_minutes)
        notify_info["dedupe_check"] = why
        if ok_to_send:
            text = _build_message(as_of, top3, n50)
            sent, reason = _telegram_send(text)
            notify_info["attempted"] = True
            notify_info["reason"] = reason
            if sent:
                _save_sent(sig, top3, as_of)
                payload["notified"] = True
            else:
                payload["notified"] = False
        else:
            notify_info["reason"] = why
            payload["notified"] = False
    else:
        if not tele_enabled:
            notify_info["reason"] = "telegram_disabled"
        elif not top3:
            notify_info["reason"] = "no_signals"
        payload["notified"] = False

    payload["notify_info"] = notify_info

    return JSONResponse(payload)
