
import os
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

# --- Imports de tu proyecto ---
# Deben existir en tu repo tal como vienen del pipeline ya funcionando
from ranking import run_full_pipeline  # -> calcula top50 y factores top3
from presend_rules import build_top3_signals  # -> aplica los 26 filtros y produce approved_top3/rejected_top3

# ------------------------------------------------------------
# Configuración
# ------------------------------------------------------------
# Token raíz fijo (como pediste): 123
RUN_TOKEN = os.getenv("RUN_TOKEN", "123")

# Directorio de trabajo seguro (escribible en Render)
DATA_DIR = Path(os.getenv("DATA_DIR", "/tmp/evolutivo"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

LAST_JSON = DATA_DIR / "last_payload.json"
LAST_SIGN = DATA_DIR / "last_signature.txt"

BOT_TOKEN = os.getenv("BOT_TOKEN")  # Telegram Bot token
CHAT_ID = os.getenv("CHAT_ID")      # Telegram chat id (grupo o usuario)
TELEGRAM_NOTIFY_EMPTY = os.getenv("TELEGRAM_NOTIFY_EMPTY", "false").lower() in ("1", "true", "yes")

# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _hash_signature(approved_top3: List[Dict[str, Any]], as_of: Optional[str]) -> str:
    # Firma estable basada en el contenido esencial de las señales aprobadas
    payload = {
        "as_of": as_of,
        "approved_top3": [
            {
                "code": s.get("code"),
                "ysymbol": s.get("ysymbol"),
                "side": s.get("side"),
                "strategy": s.get("strategy"),
                "trigger": s.get("trigger"),
                "sl": s.get("sl"),
                "tp": s.get("tp"),
                "rr": s.get("rr"),
            }
            for s in (approved_top3 or [])
        ],
    }
    b = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()

def _read_last_signature() -> Optional[str]:
    if LAST_SIGN.exists():
        try:
            return LAST_SIGN.read_text(encoding="utf-8").strip()
        except Exception:
            return None
    return None

def _write_last_signature(sig: str) -> None:
    LAST_SIGN.write_text(sig, encoding="utf-8")

def _persist_payload(payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload["_persisted_at"] = _now_iso()
    with LAST_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _load_last_payload() -> Optional[Dict[str, Any]]:
    if not LAST_JSON.exists():
        return None
    try:
        with LAST_JSON.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _escape_html(text: str) -> str:
    # Telegram parse_mode=HTML requiere escapar & < >
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )

def _fmt_price(v: Any) -> str:
    try:
        if v is None:
            return "—"
        return f"{float(v):.4f}"
    except Exception:
        return str(v)

def _fmt_rr(v: Any) -> str:
    try:
        if v is None:
            return "—"
        return f"{float(v):.1f}"
    except Exception:
        return str(v)

def _format_tg_message(payload: Dict[str, Any]) -> str:
    as_of = payload.get("as_of") or _now_iso()[:10]
    approved = payload.get("approved_top3") or []

    header = f"<b>Top 3 Señales</b>  <i>{_escape_html(as_of)}</i>\n"
    if not approved:
        body = "Sin señales aprobadas."
        return header + body

    lines = [header]
    for i, s in enumerate(approved, 1):
        code = _escape_html(s.get("code") or s.get("ysymbol") or "?")
        side = _escape_html(s.get("side") or "?")
        strat = _escape_html(s.get("strategy") or "?")
        trig = _fmt_price(s.get("trigger"))
        sl   = _fmt_price(s.get("sl"))
        tp   = _fmt_price(s.get("tp"))
        rr   = _fmt_rr(s.get("rr"))
        atr  = _fmt_price(s.get("atr")) if "atr" in s else "—"

        lines.append(
            f"<b>{i}. {code}</b>  •  {side}  •  {strat}\n"
            f"Trigger: <code>{trig}</code>  |  SL: <code>{sl}</code>  |  TP: <code>{tp}</code>  |  RR: <code>{rr}</code>\n"
            f"ATR: <code>{atr}</code>"
        )

    # Preview Top50 (opcional, primeros 5)
    top50 = payload.get("top50") or []
    if top50:
        preview = ", ".join(_escape_html(x.get("code") or x.get("ysymbol") or "?") for x in top50[:5])
        lines.append(f"\n<b>Top50 (preview)</b>: {preview}")

    return "\n".join(lines)

# Envío robusto a Telegram con fallback (requests -> httpx -> urllib)
def _send_tg(text: str) -> Tuple[bool, str]:
    if not BOT_TOKEN or not CHAT_ID:
        return False, "BOT_TOKEN/CHAT_ID no configurados"
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    # Try requests
    try:
        import requests  # type: ignore
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code == 200:
            return True, "ok"
        return False, f"HTTP {r.status_code}: {r.text}"
    except Exception as e:
        last_err = f"requests: {e!r}"

    # Try httpx
    try:
        import httpx  # type: ignore
        with httpx.Client(timeout=15.0) as client:
            r = client.post(url, json=payload)
            if r.status_code == 200:
                return True, "ok"
            return False, f"httpx HTTP {r.status_code}: {r.text}"
    except Exception as e:
        last_err = f"httpx: {e!r}"

    # Fallback urllib
    try:
        import urllib.request
        import urllib.error

        req = urllib.request.Request(url, method="POST")
        req.add_header("Content-Type", "application/json")
        data = json.dumps(payload).encode("utf-8")
        with urllib.request.urlopen(req, data, timeout=15) as resp:
            if resp.status == 200:
                return True, "ok"
            else:
                return False, f"urllib HTTP {resp.status}"
    except Exception as e:
        last_err = f"urllib: {e!r}"

    return False, last_err

app = FastAPI(title="Evolutivo Signals API", version="1.0.0")

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "telegram_enabled": bool(BOT_TOKEN and CHAT_ID),
        "data_dir": str(DATA_DIR),
        "has_last": LAST_JSON.exists(),
        "time": _now_iso(),
    }

@app.get("/signals/status")
def signals_status():
    last = _load_last_payload()
    return {
        "ok": True,
        "has_last": last is not None,
        "last_as_of": (last or {}).get("as_of"),
        "telegram_enabled": bool(BOT_TOKEN and CHAT_ID),
    }

@app.get("/signals/top3")
def get_last_top3():
    last = _load_last_payload()
    if not last:
        raise HTTPException(404, "Aún no hay resultados. Ejecuta /signals/run-top3")
    return JSONResponse(last)

# --- Endpoint ÚNICO que ejecuta TODO el análisis y (opcional) publica en Telegram ---
@app.get("/signals/run-top3")
def run_all(
    token: str = Query(..., description="Root token (p.ej. 123)"),
    send_tg: bool = Query(True, description="Si True, envía a Telegram al finalizar"),
    dedupe_tg: bool = Query(True, description="Evita duplicados (no envía si no hay cambios)")
):
    if token != RUN_TOKEN:
        raise HTTPException(status_code=403, detail="Token inválido")

    # 1) Ejecutar pipeline (Top50 + factores Top3)
    payload = run_full_pipeline()  # dict con keys: top50, top3_factors, as_of, etc.

    # 2) Construir señales finales (aplican 26 filtros internos)
    top3_factors = payload.get("top3_factors") or []
    approved_top3, rejected_top3 = build_top3_signals(top3_factors, as_of=payload.get("as_of"))

    payload["approved_top3"] = approved_top3
    payload["rejected_top3"] = rejected_top3

    # 3) Persistir payload
    _persist_payload(payload)

    # 4) Deduplicación / Envío a Telegram
    tg_result = {"attempted": False, "sent": False, "reason": "send_tg=False"}
    if send_tg:
        # Si no hay aprobadas y la política es no notificar vacío, saltamos
        if not approved_top3 and not TELEGRAM_NOTIFY_EMPTY:
            tg_result = {"attempted": True, "sent": False, "reason": "sin señales aprobadas"}
        else:
            new_sig = _hash_signature(approved_top3, payload.get("as_of"))
            old_sig = _read_last_signature()
            if dedupe_tg and old_sig and old_sig == new_sig:
                tg_result = {"attempted": True, "sent": False, "reason": "duplicado (sin cambios)"}
            else:
                msg = _format_tg_message(payload)
                ok, info = _send_tg(msg)
                if ok:
                    _write_last_signature(new_sig)
                    tg_result = {"attempted": True, "sent": True, "info": info}
                else:
                    tg_result = {"attempted": True, "sent": False, "error": info}

    out = dict(payload)
    out["telegram"] = tg_result
    return JSONResponse(out)

# --- Alias retro-compatibles (no los uses si no los necesitas) ---
@app.get("/rank/run-top3")
def alias_rank_run_top3(
    token: str = Query(...),
    send_tg: bool = Query(True),
    dedupe_tg: bool = Query(True)
):
    return run_all(token=token, send_tg=send_tg, dedupe_tg=dedupe_tg)

# Mantener alias antiguo si algún dashboard lo llama
@app.get("/rank/run")
def old_rank_run(token: str = Query(...), send_tg: bool = Query(True), dedupe_tg: bool = Query(True)):
    return run_all(token=token, send_tg=send_tg, dedupe_tg=dedupe_tg)
