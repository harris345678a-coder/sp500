
# app_evolutivo.py
# FastAPI app para ejecutar el pipeline de ranking y notificar el TOP3 a Telegram
# Diseño robusto: sin dependencias externas raras para el envío (usa requests si está,
# y si no, fallback a urllib). Evita errores de "parse entities" en Telegram enviando
# TEXTO PLANO por defecto y con reintentos controlados.

import os
import json
import time
import hashlib
import logging
from typing import Any, Dict, List, Optional

# --- Logging base (silencioso por defecto) ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("app_evolutivo")

# --- Carga módulo de ranking ---
try:
    import ranking  # se asume 'ranking.py' está junto al app
except Exception as e:
    log.exception("No se pudo importar 'ranking'.")
    raise

# --- FastAPI app ---
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import JSONResponse
except Exception as e:
    log.exception("FastAPI no está disponible en el entorno.")
    raise

app = FastAPI(title="App Evolutivo", version="1.0.0")

# ---- Utilidades generales ----


def _dedupe_ordered(seq: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
def _now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _digest_for(items: List[str]) -> str:
    msg = "|".join(items).encode("utf-8")
    return hashlib.sha256(msg).hexdigest()

def _read_last_digest(state_dir: str) -> Optional[str]:
    try:
        p = os.path.join(state_dir, "last_top3_digest.txt")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as fh:
                return fh.read().strip()
    except Exception:
        pass
    return None

def _write_last_digest(state_dir: str, digest: str) -> None:
    try:
        os.makedirs(state_dir, exist_ok=True)
        p = os.path.join(state_dir, "last_top3_digest.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(digest)
    except Exception:
        log.warning("No pude persistir el digest en %s", state_dir)

# ---- Telegram ----

def _http_post(url: str, data: Dict[str, Any]) -> Dict[str, Any]:
    # Intenta con requests primero, luego urllib.
    try:
        import requests  # type: ignore
        resp = requests.post(url, data=data, timeout=10)
        return {"status": resp.status_code, "text": resp.text}
    except Exception as e:
        import urllib.request
        import urllib.parse
        try:
            encoded = urllib.parse.urlencode(data).encode("utf-8")
            with urllib.request.urlopen(url, data=encoded, timeout=10) as resp:
                body = resp.read().decode("utf-8")
                return {"status": getattr(resp, "status", 200), "text": body}
        except Exception as e2:
            return {"status": 0, "text": f"exception: {e!r} / {e2!r}"}

def _escape_markdown_v2(s: str) -> str:
    # Escapa todos los caracteres especiales de MarkdownV2 según la doc de Telegram.
    chars = r'_*[]()~`>#+-=|{}.!'
    out = []
    for ch in s:
        if ch in chars:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)

def _build_top3_message(as_of: str, top3: List[Dict[str, Any]], diag: Dict[str, Any]) -> str:
    # Mensaje en TEXTO PLANO. Evita parse_mode para impedir errores de "parse entities".
    lines = []
    lines.append(f"TOP 3 — {as_of}")
    for i, item in enumerate(top3, start=1):
        tk = item.get("ticker", "?")
        reasons = item.get("reasons", [])
        rs = ", ".join(map(str, reasons))
        lines.append(f"{i}) {tk} — {rs}")
    # Datos de universo si están disponibles
    ucount = diag.get("universe_count")
    fcount = diag.get("fetched_count")
    excount = diag.get("excluded_count")
    if any(x is not None for x in [ucount, fcount, excount]):
        lines.append("")
        details = []
        if ucount is not None: details.append(f"universo={ucount}")
        if fcount is not None: details.append(f"fetched={fcount}")
        if excount is not None: details.append(f"excluidos={excount}")
        lines.append(" | ".join(details))
    return "\n".join(lines)

def send_telegram_top3(as_of: str, top3: List[Dict[str, Any]], diag: Dict[str, Any]) -> Dict[str, Any]:
    info: Dict[str, Any] = {"attempted": False, "enabled": False}
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    enabled = os.getenv("TELEGRAM_ENABLED", "1").strip() not in ("0", "false", "False", "")
    info["enabled"] = enabled

    if not enabled:
        info["reason"] = "disabled_by_env"
        return info

    if not token or not chat_id:
        info["attempted"] = False
        info["reason"] = "missing_env(TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID)"
        return info

    # Solo intentamos enviar si hay al menos 1 señal
    if not top3:
        info["attempted"] = False
        info["reason"] = "no_top3"
        return info

    # Dedupe por digest en un directorio de estado
    state_dir = os.getenv("STATE_DIR", "/tmp")
    tickers = [str(x.get("ticker", "")) for x in top3]
    digest = _digest_for(tickers)
    info["tickers"] = tickers

    last = _read_last_digest(state_dir)
    if last == digest:
        info["attempted"] = False
        info["reason"] = "duplicate_already_sent"
        info["dedupe_check"] = "duplicate_blocked"
        return info
    info["dedupe_check"] = "ok_to_send"

    # Construye texto (PLANO por defecto)
    text_plain = _build_top3_message(as_of, top3, diag)

    # Primer intento: TEXTO PLANO sin parse_mode (robusto)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text_plain,
        "disable_web_page_preview": True,
    }

    info["attempted"] = True
    resp = _http_post(url, payload)
    status = resp.get("status", 0)
    body = resp.get("text", "")

    if status == 200:
        _write_last_digest(state_dir, digest)
        info["status"] = status
        info["response"] = "ok"
        return info

    # Si falló por parseo (raro al ser texto plano), reintenta con MarkdownV2 escapado
    parse_error_mark = "can't parse entities"
    if parse_error_mark in body.lower():
        escaped = _escape_markdown_v2(text_plain)
        payload_md = {
            "chat_id": chat_id,
            "text": escaped,
            "parse_mode": "MarkdownV2",
            "disable_web_page_preview": True,
        }
        resp2 = _http_post(url, payload_md)
        status2 = resp2.get("status", 0)
        body2 = resp2.get("text", "")
        if status2 == 200:
            _write_last_digest(state_dir, digest)
            info["status"] = status2
            info["response"] = "ok_after_escape"
            return info
        info["status"] = status2
        info["response"] = f"send_failed_after_escape({status2} {body2})"
        return info

    info["status"] = status
    info["response"] = f"send_failed(status={status} body={body})"
    return info

# ---- Endpoints ----

@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"ok": True, "ts": _now_iso()}

@app.get("/rank/run-top3")
def run_top3(token: Optional[str] = Query(default=None)) -> JSONResponse:
    # Autorización básica por token (opcional)
    required = os.getenv("API_TOKEN", "123")
    if required and (token or "") != required:
        raise HTTPException(status_code=401, detail="invalid token")

    t0 = time.time()
    result: Dict[str, Any] = {}

    # Ejecuta pipeline sin kwargs no soportados (robusto vs cambios de firma)
    try:
        payload = ranking.run_full_pipeline()  # debe devolver dict con top50/top3_factors/diag/...
    except TypeError as e:
        # Si el proyecto viejo esperara kwargs, reintenta con detección
        log.warning("run_full_pipeline() TypeError: %r", e)
        try:
            payload = ranking.run_full_pipeline()  # llamada simple
        except Exception as e2:
            log.exception("Fallo irrecuperable ejecutando run_full_pipeline().")
            raise HTTPException(status_code=500, detail=f"pipeline_error: {e2!r}")
    except Exception as e:
        log.exception("Fallo ejecutando run_full_pipeline().")
        raise HTTPException(status_code=500, detail=f"pipeline_error: {e!r}")

    as_of = str(payload.get("as_of", ""))
    top50 = payload.get("top50", []) or []
    # dedupe preservando orden, sin alterar el payload original
    top50 = _dedupe_ordered(top50)
    top3 = payload.get("top3_factors", []) or []
    # dedupe por ticker en top3
    _seen_tk = set()
    _top3_dedup = []
    for item in top3:
        tk = str(item.get("ticker", ""))
        if tk and tk not in _seen_tk:
            _seen_tk.add(tk)
            _top3_dedup.append(item)
    top3 = _top3_dedup
    diag  = payload.get("diag", {}) or {}

    # Notificar a Telegram (robusto)
    notify_info = send_telegram_top3(as_of, top3, diag)
    notified = notify_info.get("response", "").startswith("ok")

    t1 = time.time()
    result = {
        "ok": True,
        "took_s": round(t1 - t0, 2),
        "as_of": as_of,
        "top50": top50[:50],  # garantizar máx 50
        "top3_factors": top3[:3],
        "diag": diag,
        "notified": notified,
        "notify_info": notify_info,
    }
    return JSONResponse(result)
