
# app_evolutivo.py
# FastAPI app with robust diagnostics, Telegram notifications, and dedup logic.
# Non-invasive: preserves existing external contracts while adding detailed logging.
from __future__ import annotations

import os
import json
import logging
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, PlainTextResponse

# ---- Logging config ---------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper().strip()
DEBUG_FLAG = os.getenv("DEBUG", "0").strip() in {"1", "true", "TRUE", "yes", "on"}
level = getattr(logging, LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("app_evolutivo")
if DEBUG_FLAG and level > logging.DEBUG:
    log.setLevel(logging.DEBUG)

# ---- Storage paths ----------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "/tmp/evolutivo"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
LAST_PAYLOAD = DATA_DIR / "last_payload.json"
LAST_SIGNATURE = DATA_DIR / "last_signature.txt"

# ---- Imports from local pipeline -------------------------------------------
# We expect your repo to have ranking.py that exposes run_full_pipeline().
# No changes to ranking.py are required for this file to work.
try:
    from ranking import run_full_pipeline  # type: ignore
except Exception as e:
    log.exception("No se pudo importar run_full_pipeline desde ranking.py")
    raise

# ---- Utils -----------------------------------------------------------------
def _json_default(o: Any):
    try:
        import numpy as np  # type: ignore
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
    except Exception:
        pass
    # Timestamps or others
    try:
        import pandas as pd  # type: ignore
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
    except Exception:
        pass
    return str(o)

def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=_json_default)

def _signature_for_payload(payload: Dict[str, Any]) -> str:
    # Dedup en base a señales aprobadas + fecha
    as_of = str(payload.get("as_of", ""))
    approved = payload.get("approved_top3") or []
    key = dumps({"as_of": as_of, "approved_top3": approved})
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")

def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None

def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )

def _telegram_enabled() -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID")
    return bool(token and chat)

def _send_telegram(msg: str, silent: bool = False) -> Dict[str, Any]:
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN") or "").strip()
    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID") or "").strip()
    resp: Dict[str, Any] = {"attempted": True, "sent": False}
    if not token or not chat_id:
        resp["reason"] = "faltan TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID"
        return resp

    payload = {
        "chat_id": chat_id,
        "text": msg,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
        "disable_notification": silent,
    }

    # Try requests -> httpx -> urllib
    try:
        import requests  # type: ignore

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, json=payload, timeout=10)
        ok = False
        try:
            data = r.json()
        except Exception:
            data = {"raw": r.text}
        if r.status_code == 200 and isinstance(data, dict) and data.get("ok"):
            ok = True
        resp["status_code"] = r.status_code
        resp["response"] = data
        resp["sent"] = ok
        if not ok:
            resp["reason"] = f"HTTP {r.status_code}"
        return resp
    except Exception as e:
        resp["requests_error"] = str(e)

    try:
        import httpx  # type: ignore

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        with httpx.Client(timeout=10) as client:
            r = client.post(url, json=payload)
            data = r.json()
            ok = r.status_code == 200 and data.get("ok") is True
            resp["status_code"] = r.status_code
            resp["response"] = data
            resp["sent"] = ok
            if not ok:
                resp["reason"] = f"HTTP {r.status_code}"
            return resp
    except Exception as e:
        resp["httpx_error"] = str(e)

    # urllib fallback
    try:
        import urllib.request
        import urllib.error

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        req = urllib.request.Request(
            url,
            data=dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as f:
            raw = f.read().decode("utf-8", "ignore")
        data = json.loads(raw)
        ok = bool(data.get("ok"))
        resp["status_code"] = 200 if ok else 500
        resp["response"] = data
        resp["sent"] = ok
        if not ok:
            resp["reason"] = "respuesta no OK"
        return resp
    except Exception as e:
        resp["urllib_error"] = str(e)
        resp["reason"] = "fallaron los 3 clientes HTTP"
        return resp

def _format_telegram(payload: Dict[str, Any]) -> str:
    as_of = payload.get("as_of", "")
    approved: List[Dict[str, Any]] = payload.get("approved_top3") or []
    top3_factors: List[Dict[str, Any]] = payload.get("top3_factors") or []
    if not approved:
        return f"<b>Sin señales aprobadas</b>\n<code>as_of={_escape_html(str(as_of))}</code>"

    lines = [f"<b>Señales aprobadas</b> <code>{_escape_html(str(as_of))}</code>"]
    for s in approved:
        code = s.get("code") or s.get("ysymbol")
        side = s.get("side")
        strat = s.get("strategy")
        trig = s.get("trigger")
        sl = s.get("sl")
        tp = s.get("tp")
        rr = s.get("rr")
        atr = s.get("atr")
        line = f"• <b>{_escape_html(str(code))}</b> {side or ''} · {strat or ''}"
        if trig is not None:
            line += f" · trg={trig}"
        if sl is not None and tp is not None:
            line += f" · SL={sl} · TP={tp}"
        if rr is not None:
            line += f" · RR={rr}"
        if atr is not None:
            line += f" · ATR={atr}"
        lines.append(line)
    # breve factor resumen si existe
    if top3_factors:
        lines.append("—")
        for f in top3_factors[:3]:
            code = f.get("code") or f.get("ysymbol")
            rsi15 = f.get("rsi15")
            rsi60 = f.get("rsi60")
            rsi4h = f.get("rsi4h")
            rsi1d = f.get("rsi1d")
            lines.append(
                f"{_escape_html(str(code))}: RSI15={rsi15} · 1H={rsi60} · 4H={rsi4h} · 1D={rsi1d}"
            )
    return "\n".join(lines)

def _diagnose(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Construye un bloque de diagnóstico explicando "por qué vacío" y el histograma de motivos
    top50 = payload.get("top50") or []
    top3_factors = payload.get("top3_factors") or []
    approved = payload.get("approved_top3") or []
    rejected = payload.get("rejected_top3") or []

    reasons_hist: Dict[str, int] = {}
    for r in rejected:
        reason = r.get("reason") or "sin_motivo"
        reasons_hist[reason] = reasons_hist.get(reason, 0) + 1

    why_empty = None
    if not approved:
        if not top3_factors and not top50:
            why_empty = "pipeline_vacio: sin top50 ni top3_factors"
        elif not top3_factors and top50:
            why_empty = "sin_top3_factors: top50 presente"
        elif top3_factors and not rejected:
            why_empty = "sin_aprobadas: top3_factors presente pero ninguna aprobada"
        else:
            # Deduce el mayor motivo de rechazo
            if reasons_hist:
                max_reason = max(reasons_hist, key=lambda k: reasons_hist[k])
                why_empty = f"rechazos_dominantes: {max_reason}"
            else:
                why_empty = "sin_aprobadas: motivos no informados"

    diag = {
        "counts": {
            "top50": len(top50),
            "top3_factors": len(top3_factors),
            "approved_top3": len(approved),
            "rejected_top3": len(rejected),
        },
        "reasons_hist": reasons_hist,
        "why_empty": why_empty,
    }
    return diag

def _persist_payload(payload: Dict[str, Any]) -> None:
    try:
        _write_text(LAST_PAYLOAD, dumps(payload))
    except Exception as e:
        log.warning("No se pudo escribir last_payload.json: %s", e)

# ---- FastAPI app ------------------------------------------------------------
app = FastAPI(title="Evolutivo Signals API", version="1.0.0")

@app.get("/", response_class=PlainTextResponse)
def root() -> str:
    return "Evolutivo · use /rank/run-top3?token=123"

def _compute_and_finalize_payload() -> Dict[str, Any]:
    # Llama a tu pipeline y devuelve el payload. Añade campos de debug si procede.
    t0 = time.time()
    payload: Dict[str, Any] = run_full_pipeline()
    t1 = time.time()
    diag = _diagnose(payload)

    log.info(
        "Pipeline listo · as_of=%s · top50=%d · top3_factors=%d · approved=%d · rejected=%d · %.2fs",
        payload.get("as_of"),
        diag["counts"]["top50"],
        diag["counts"]["top3_factors"],
        diag["counts"]["approved_top3"],
        diag["counts"]["rejected_top3"],
        (t1 - t0),
    )
    if diag["reasons_hist"]:
        log.debug("Histograma de rechazos: %s", diag["reasons_hist"])
    if diag["why_empty"]:
        log.debug("Diagnóstico vacío: %s", diag["why_empty"])

    if DEBUG_FLAG or logging.getLogger().level <= logging.DEBUG:
        payload.setdefault("debug", {})
        payload["debug"]["diagnostics"] = diag

    _persist_payload(payload)
    return payload

def _maybe_send_telegram(payload: Dict[str, Any]) -> Dict[str, Any]:
    approved = payload.get("approved_top3") or []
    if not _telegram_enabled():
        return {"attempted": False, "sent": False, "reason": "telegram_desactivado"}

    # Deduplicación por firma
    sig = _signature_for_payload(payload)
    prev = _read_text(LAST_SIGNATURE)

    if prev == sig:
        log.debug("Telegram dedup: misma firma, no se envía")
        info = {"attempted": True, "sent": False, "reason": "duplicado (sin cambios)"}
        payload.setdefault("telegram", info)
        return info

    msg = _format_telegram(payload)
    resp = _send_telegram(msg, silent=(not approved))
    if resp.get("sent"):
        _write_text(LAST_SIGNATURE, sig)
        log.info("Telegram enviado OK")
    else:
        log.warning("Telegram falló: %s", resp.get("reason"))
    payload.setdefault("telegram", resp)
    return resp

def signals_run(token: str) -> JSONResponse:
    # Token es pasivo (backwards compat); no imponemos auth para no romper integración
    try:
        payload = _compute_and_finalize_payload()
    except Exception as e:
        log.exception("Fallo ejecutando pipeline")
        return JSONResponse(
            {"ok": False, "error": "pipeline_error", "detail": str(e)},
            status_code=500,
        )

    try:
        _maybe_send_telegram(payload)
    except Exception as e:
        log.exception("Fallo enviando Telegram")
        payload.setdefault("telegram", {"attempted": True, "sent": False, "reason": f"error: {e}"})

    return JSONResponse(payload)

# ---- Routes (aliases conservando compat) -----------------------------------
@app.get("/signals/run-top3")
def api_signals_run(token: str = Query(default="123")):
    return signals_run(token)

@app.get("/rank/run-top3")
def old_rank_run(token: str = Query(default="123")):
    return signals_run(token)

@app.get("/rank/run")
def rank_run(token: str = Query(default="123")):
    return signals_run(token)

@app.get("/debug/last")
def debug_last():
    if LAST_PAYLOAD.exists():
        try:
            raw = LAST_PAYLOAD.read_text(encoding="utf-8")
            return JSONResponse(json.loads(raw))
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return JSONResponse({"ok": False, "error": "no_last_payload"}, status_code=404)

@app.get("/healthz", response_class=PlainTextResponse)
def healthz() -> str:
    return "ok"
