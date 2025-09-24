
import os
from pathlib import Path
import json
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

# --- Tu pipeline existente ---
# Deben existir en el repo (¡no los tocamos aquí!)
from ranking import run_full_pipeline     # -> calcula top50 y factores top3
from presend_rules import build_top3_signals  # -> aplica TODOS los filtros (26) y produce señales finales

# ============================================================
# Configuración
# ============================================================
RUN_TOKEN = os.getenv("RUN_TOKEN", "123")  # token raíz (fijo por tu requerimiento)
DATA_DIR = Path(os.getenv("DATA_DIR", "/tmp/evolutivo"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

LAST_JSON = DATA_DIR / "last_payload.json"
LAST_SIGN = DATA_DIR / "last_signature.txt"

BOT_TOKEN = os.getenv("BOT_TOKEN")  # Telegram Bot token
CHAT_ID   = os.getenv("CHAT_ID")    # Telegram chat id (grupo o usuario)
TELEGRAM_NOTIFY_EMPTY = os.getenv("TELEGRAM_NOTIFY_EMPTY", "false").lower() in ("1","true","yes")

# ============================================================
# Utilidades
# ============================================================
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

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

def _write_last_signature(sig: str) -> None:
    LAST_SIGN.write_text(sig, encoding="utf-8")

def _read_last_signature() -> Optional[str]:
    try:
        return LAST_SIGN.read_text(encoding="utf-8").strip() if LAST_SIGN.exists() else None
    except Exception:
        return None

def _hash_signature(approved_top3: List[Dict[str, Any]], as_of: Optional[str]) -> str:
    # Firma estable basada en campos esenciales para evitar duplicados
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
                "tp1": s.get("tp1"),
                "tp2": s.get("tp2"),
                "rr": s.get("rr"),
            }
            for s in (approved_top3 or [])
        ],
    }
    b = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()

# ============================================================
# Enriquecimiento de niveles: TG, SL, TP1, TP2
# ============================================================
def _enrich_targets(s: Dict[str, Any]) -> Dict[str, Any]:
    """
    Asegura que cada señal tenga: trigger (TG), SL, TP1, TP2.
    - Si ya vienen de presend_rules, se respetan.
    - Si falta TP1/TP2, se calculan usando ATR y el trigger como entrada.
    - Fórmula por defecto (simétrica y robusta):
        LONG:  SL = TG - 1.20*ATR, TP1 = TG + 1.00*ATR, TP2 = TG + 1.90*ATR
        SHORT: SL = TG + 1.20*ATR, TP1 = TG - 1.00*ATR, TP2 = TG - 1.90*ATR
    - RR se mantiene tal como venga (basado en TP2), y si falta se calcula contra TP2.
    """
    out = dict(s)
    side = (out.get("side") or "").lower().strip()
    atr  = out.get("atr")
    tg   = out.get("trigger")

    # Si no hay trigger, usamos price si viene; si no, lo dejamos None
    price = out.get("price") or out.get("entry") or out.get("close")
    if tg is None:
        tg = price

    # Si falta ATR pero hay SL/TP, tratamos de inferir uno aproximado
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    tg_f  = _to_float(tg)
    sl_f  = _to_float(out.get("sl"))
    tp_f  = _to_float(out.get("tp"))  # algunos pipelines traen 'tp' único (lo tomamos como TP2)
    tp1_f = _to_float(out.get("tp1"))
    tp2_f = _to_float(out.get("tp2"))
    atr_f = _to_float(atr)

    # Si no hay ATR pero sí TG y SL o TP, inferimos ATR aproximado
    if atr_f is None and tg_f is not None:
        if sl_f is not None:
            atr_f = abs(tg_f - sl_f) / 1.20
        elif tp_f is not None:
            atr_f = abs(tp_f - tg_f) / 1.90

    # Calculamos TP1/TP2 si faltan y hay side + TG + ATR
    if tp1_f is None or tp2_f is None:
        if side in ("long", "short") and tg_f is not None and atr_f is not None and atr_f > 0:
            if side == "long":
                tp1_f = tg_f + 1.00 * atr_f if tp1_f is None else tp1_f
                tp2_f = tg_f + 1.90 * atr_f if tp2_f is None else tp2_f
                sl_f  = tg_f - 1.20 * atr_f if sl_f  is None else sl_f
            else:
                tp1_f = tg_f - 1.00 * atr_f if tp1_f is None else tp1_f
                tp2_f = tg_f - 1.90 * atr_f if tp2_f is None else tp2_f
                sl_f  = tg_f + 1.20 * atr_f if sl_f  is None else sl_f

    # Si no venía RR, lo calculamos respecto a TP2
    rr_f = _to_float(out.get("rr"))
    if rr_f is None and tg_f is not None and sl_f is not None and tp2_f is not None and (tg_f != sl_f):
        rr_f = abs((tp2_f - tg_f) / (tg_f - sl_f))

    # Asignamos salidas formateadas
    out["trigger"] = tg_f if tg_f is not None else out.get("trigger")
    out["sl"] = sl_f if sl_f is not None else out.get("sl")
    out["tp1"] = tp1_f if tp1_f is not None else out.get("tp1")
    out["tp2"] = tp2_f if tp2_f is not None else (tp_f if tp_f is not None else out.get("tp2"))
    out["rr"] = rr_f if rr_f is not None else out.get("rr")
    out["atr"] = atr_f if atr_f is not None else out.get("atr")
    return out

def _enrich_all(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_enrich_targets(s) for s in (signals or [])]

# ============================================================
# Mensaje para Telegram (HTML)
# ============================================================
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
        tp1  = _fmt_price(s.get("tp1"))
        tp2  = _fmt_price(s.get("tp2"))
        rr   = _fmt_rr(s.get("rr"))
        atr  = _fmt_price(s.get("atr")) if ("atr" in s) else "—"

        lines.append(
            f"<b>{i}. {code}</b>  •  {side}  •  {strat}\n"
            f"TG: <code>{trig}</code>  |  SL: <code>{sl}</code>  |  TP1: <code>{tp1}</code>  |  TP2: <code>{tp2}</code>  |  RR: <code>{rr}</code>\n"
            f"ATR: <code>{atr}</code>"
        )

    # Preview Top50 (primeros 5) – opcional
    top50 = payload.get("top50") or []
    if top50:
        def _code_of(x):
            return _escape_html((x.get("code") or x.get("ysymbol") or x.get("symbol") or "?"))
        preview = ", ".join(_code_of(x) for x in top50[:5])
        lines.append(f"\n<b>Top50 (preview)</b>: {preview}")

    return "\n".join(lines)

# ============================================================
# Envío robusto a Telegram (requests -> httpx -> urllib)
# ============================================================
def _send_tg(text: str) -> Tuple[bool, str]:
    if not BOT_TOKEN or not CHAT_ID:
        return False, "missing_env(TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID)"
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    last_err = "unknown"
    # Try requests
    try:
        import requests  # type: ignore
        r = requests.post(url, json=payload, timeout=20)
        if r.status_code == 200:
            return True, "ok"
        return False, f"send_failed(status={r.status_code} body={r.text})"
    except Exception as e:
        last_err = f"requests: {e!r}"

    # Try httpx
    try:
        import httpx  # type: ignore
        with httpx.Client(timeout=20.0) as client:
            r = client.post(url, json=payload)
        if r.status_code == 200:
            return True, "ok"
        return False, f"send_failed(status={r.status_code} body={r.text})"
    except Exception as e:
        last_err = f"httpx: {e!r}"

    # Fallback urllib
    try:
        import urllib.request
        import urllib.error
        req = urllib.request.Request(url, method="POST")
        req.add_header("Content-Type", "application/json")
        data = json.dumps(payload).encode("utf-8")
        with urllib.request.urlopen(req, data, timeout=20) as resp:
            if resp.status == 200:
                return True, "ok"
            return False, f"urllib status={resp.status}"
    except Exception as e:
        last_err = f"urllib: {e!r}"

    return False, last_err

# ============================================================
# FastAPI
# ============================================================
app = FastAPI(title="Evolutivo Signals API", version="1.1.0")

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

# --- ENDPOINT ÚNICO: ejecuta TODO y (opcional) envía Telegram ---
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

    # 2) Construir señales finales (aplican TODOS los filtros internos)
    top3_factors = payload.get("top3_factors") or []
    approved_top3, rejected_top3 = build_top3_signals(top3_factors, as_of=payload.get("as_of"))

    # 3) Enriquecer con TG/SL/TP1/TP2 y recalcular RR si falta
    approved_top3 = _enrich_all(approved_top3)

    # 4) Armar payload final y persistir
    payload["approved_top3"] = approved_top3
    payload["rejected_top3"] = rejected_top3
    _persist_payload(payload)

    # 5) Notificación (opcional) — SIEMPRE después de TODOS los filtros
    notified = False
    notify_info: Dict[str, Any] = {"attempted": False, "enabled": bool(BOT_TOKEN and CHAT_ID), "tickers": [s.get("code") or s.get("ysymbol") for s in approved_top3]}

    if send_tg:
        notify_info["attempted"] = True

        # Dedupe por firma del contenido (evita repetir si no hubo cambios)
        sig = _hash_signature(approved_top3, payload.get("as_of"))
        last_sig = _read_last_signature()

        if dedupe_tg and last_sig == sig:
            notify_info.update({"dedupe_check": "duplicate_skipped"})
        else:
            # Construir mensaje
            text = _format_tg_message(payload)

            # Si no hay aprobadas y NO queremos notificar vacíos, salteamos
            if not approved_top3 and not TELEGRAM_NOTIFY_EMPTY:
                notify_info.update({"reason": "empty_and_notify_false"})
            else:
                ok, info = _send_tg(text)
                notify_info.update({"status": 200 if ok else 500, "response": info})
                if ok:
                    notified = True
                    _write_last_signature(sig)

    return JSONResponse({
        "ok": True,
        "took_s": None,  # el timing lo maneja tu infra de logs
        "as_of": payload.get("as_of"),
        "top50": [x.get("code") or x.get("ysymbol") or x.get("symbol") for x in (payload.get("top50") or [])][:50],
        "top3_factors": payload.get("top3_factors"),
        "approved_top3": approved_top3,
        "rejected_top3": rejected_top3,
        "notified": notified,
        "notify_info": notify_info,
    })
