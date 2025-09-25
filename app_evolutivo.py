
import os
import io
import json
import math
import time
import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import requests
import yfinance as yf

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

# =========================
# Logging
# =========================
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("app_evolutivo")

# =========================
# Config
# =========================
API_TOKEN = os.getenv("API_TOKEN", "123")

BOT_TOKEN = (
    os.getenv("TELEGRAM_BOT_TOKEN")
    or os.getenv("BOT_TOKEN")
    or ""
)
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID") or ""

TELEGRAM_NOTIFY_EMPTY = os.getenv("TELEGRAM_NOTIFY_EMPTY", "false").lower() == "true"

STATE_DIR = "/tmp/evolutivo_state"
os.makedirs(STATE_DIR, exist_ok=True)
LAST_PAYLOAD_PATH = os.path.join(STATE_DIR, "last_payload.json")
LAST_DIGEST_PATH = os.path.join(STATE_DIR, "last_digest.txt")

# =========================
# Helpers
# =========================
def _escape_html(text: str) -> str:
    # Only basic escaping; we still use <b> around our header
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )

def _auth_or_403(token: Optional[str]):
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="invalid token")

def _save_json(path: str, data: Any):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        log.warning("No pude guardar JSON en %s: %s", path, e)

def _read_json(path: str) -> Optional[Any]:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("No pude leer JSON en %s: %s", path, e)
        return None

def _read_text(path: str) -> Optional[str]:
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None

def _write_text(path: str, text: str):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        log.warning("No pude guardar texto en %s: %s", path, e)

def _digest_messages(msgs: List[str]) -> str:
    joined = "\n\n".join(msgs).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()

# =========================
# Indicators (no extra deps)
# =========================
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _rma(series: pd.Series, length: int) -> pd.Series:
    # Wilder's smoothing (RMA)
    alpha = 1.0 / float(length)
    return series.ewm(alpha=alpha, adjust=False).mean()

def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return _rma(tr, length)

def _adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    plus_dm = (high - prev_high)
    minus_dm = (prev_low - low)

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = _atr(df, length) * 1.0  # reuse to get TR smoothed denominator consistently
    # NOTE: _atr already returns smoothed TR; for DI calculus we need RMA of raw TR:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr_raw = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    tr_rma = _rma(tr_raw, length)

    plus_di = 100.0 * (_rma(plus_dm, length) / tr_rma).replace([np.inf, -np.inf], np.nan)
    minus_di = 100.0 * (_rma(minus_dm, length) / tr_rma).replace([np.inf, -np.inf], np.nan)

    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).abs()).replace([np.inf, -np.inf], np.nan)
    adx = _rma(dx, length)
    return adx

def _avg_dollar_vol(df: pd.DataFrame, n: int = 20) -> float:
    dv = (df["Close"].astype(float) * df["Volume"].astype(float)).rolling(n).mean()
    val = dv.iloc[-1] if len(dv) else np.nan
    try:
        return float(val) if pd.notna(val) else 0.0
    except Exception:
        try:
            return float(val.item())
        except Exception:
            return 0.0

# =========================
# Data fetch and single-ticker analysis (fallback)
# =========================
def _analyze_one_ticker(ticker: str) -> Optional[Dict[str, Any]]:
    try:
        period = "2y"
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        if df is None or df.empty:
            return None

        close = df["Close"].astype(float)
        ema_fast = _ema(close, 20)
        ema_slow = _ema(close, 50)
        mom = close.pct_change(63) * 100.0  # 63 trading days ≈ 3 meses
        adx = _adx(df, 14)
        atr = _atr(df, 14)

        last_price = float(close.iloc[-1])
        last_ema_fast = float(ema_fast.iloc[-1])
        last_ema_slow = float(ema_slow.iloc[-1])
        last_mom = float(mom.iloc[-1]) if not math.isnan(mom.iloc[-1]) else 0.0
        last_adx = float(adx.iloc[-1]) if not math.isnan(adx.iloc[-1]) else 0.0
        last_atr = abs(float(atr.iloc[-1])) if not math.isnan(atr.iloc[-1]) else 0.0

        # Side selection similar to ranking rules
        side = "long" if (last_ema_fast > last_ema_slow and last_mom > 0 and last_adx >= 15.0) else "short"

        tg = last_price
        # Enrichment: ATR-based dynamic targets
        # LONG: SL=TG−1.20*ATR, TP1=TG+1.00*ATR, TP2=TG+1.90*ATR
        # SHORT: invert
        if side == "long":
            sl = tg - 1.20 * last_atr
            tp1 = tg + 1.00 * last_atr
            tp2 = tg + 1.90 * last_atr
        else:
            sl = tg + 1.20 * last_atr
            tp1 = tg - 1.00 * last_atr
            tp2 = tg - 1.90 * last_atr

        rr = 0.0
        risk = abs(tg - sl)
        reward = abs(tp2 - tg)
        if risk > 1e-9:
            rr = reward / risk

        return {
            "symbol": ticker,
            "code": ticker,
            "ysymbol": ticker,
            "side": side,
            "strategy": "Evolutivo/ATR",
            "tg": round(tg, 4),
            "sl": round(sl, 4),
            "tp1": round(tp1, 4),
            "tp2": round(tp2, 4),
            "rr": round(rr, 2),
            "atr14": round(last_atr, 4),
            "ema20": round(last_ema_fast, 4),
            "ema50": round(last_ema_slow, 4),
            "mom63": round(last_mom, 4),
            "adx14": round(last_adx, 2),
            "avg_dollar_vol20": round(_avg_dollar_vol(df, 20), 2),
        }
    except Exception as e:
        log.error("fallback analyze failed for %s: %s", ticker, e, exc_info=True)
        return None

# =========================
# presend_rules adapter
# =========================
def _normalize_symbolish(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        d = dict(x)
        code = d.get("code") or d.get("ysymbol") or d.get("symbol") or d.get("ticker")
        if code:
            d.setdefault("symbol", code)
            d.setdefault("code", code)
            d.setdefault("ysymbol", code)
        return d
    elif isinstance(x, str):
        return {"symbol": x, "code": x, "ysymbol": x}
    else:
        return {"symbol": str(x), "code": str(x), "ysymbol": str(x)}

def _shape_for_presend_rules(payload: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(payload)
    for key in ("top50", "top3_factors", "features", "finalists", "approved_top3"):
        if key in p and isinstance(p[key], list):
            p[key] = [_normalize_symbolish(it) for it in p[key]]
    # Some ranking outputs use 'ticker' field inside dicts; ensure code/symbol present
    return p

def _use_presend_rules(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        import presend_rules  # provided by your codebase
    except Exception as e:
        log.error("No se pudo importar presend_rules: %s", e)
        raise

    try:
        p = _shape_for_presend_rules(payload)
        out = presend_rules.build_top3_signals(p)  # type: ignore
        # Expect out to be a list of dicts w/ levels; if strings slipped in, normalize
        if isinstance(out, list):
            out = [_normalize_symbolish(it) for it in out]
        return out
    except Exception as e:
        log.error("Error en presend_rules.build_top3_signals", exc_info=True)
        raise

# =========================
# Telegram
# =========================
def _format_signal_msg(sig: Dict[str, Any]) -> str:
    sym = sig.get("symbol") or sig.get("code") or sig.get("ysymbol") or "?"
    side = (sig.get("side") or "").upper()
    strat = sig.get("strategy") or "Evolutivo"

    tg = sig.get("tg"); sl = sig.get("sl"); tp1 = sig.get("tp1"); tp2 = sig.get("tp2")
    rr = sig.get("rr"); atr = sig.get("atr14") or sig.get("atr") or sig.get("atr_14")

    header = f"<b>{_escape_html(str(sym))} • { _escape_html(side) } • { _escape_html(str(strat)) }</b>"
    lines = [
        header,
        f"TG: {tg} | SL: {sl} | TP1: {tp1} | TP2: {tp2} | RR: {rr} | ATR14: {atr}"
    ]
    return "\n".join(lines)

def _send_tg_messages(msgs: List[str], dedupe: bool = True) -> Dict[str, Any]:
    if not BOT_TOKEN or not CHAT_ID:
        return {"sent": False, "reason": "telegram_not_configured"}

    if not msgs:
        if TELEGRAM_NOTIFY_EMPTY:
            msgs = ["<b>Señales</b>\nSin señales aprobadas."]
        else:
            return {"sent": False, "reason": "empty_and_notify_disabled"}

    if dedupe:
        new_digest = _digest_messages(msgs)
        last_digest = _read_text(LAST_DIGEST_PATH)
        if last_digest and last_digest == new_digest:
            return {"sent": False, "reason": "duplicate_digest"}
    else:
        new_digest = None

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    ok_count = 0
    for m in msgs:
        payload = {
            "chat_id": CHAT_ID,
            "text": m,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            r = requests.post(url, json=payload, timeout=12)
            if r.ok:
                ok_count += 1
            else:
                log.warning("Telegram no OK: %s %s", r.status_code, r.text)
        except Exception as e:
            log.warning("Telegram fallo: %s", e)

    if dedupe and new_digest:
        _write_text(LAST_DIGEST_PATH, new_digest)

    return {"sent": ok_count > 0, "count": ok_count}

# =========================
# Orchestrator
# =========================
def _build_messages_from_signals(signals: List[Dict[str, Any]]) -> List[str]:
    out = []
    for s in signals:
        try:
            out.append(_format_signal_msg(s))
        except Exception as e:
            log.warning("No pude formatear una señal: %s", e)
    return out

def _fallback_build_signals(tickers: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for tk in tickers:
        res = _analyze_one_ticker(tk)
        if res:
            out.append(res)
    return out

def _persist_last_payload(payload: Dict[str, Any]):
    _save_json(LAST_PAYLOAD_PATH, payload)

# =========================
# FastAPI
# =========================
app = FastAPI()

@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "now": datetime.utcnow().isoformat(),
        "telegram": bool(BOT_TOKEN and CHAT_ID),
        "notify_empty": TELEGRAM_NOTIFY_EMPTY,
    }

@app.get("/signals/status")
def signals_status():
    last = _read_json(LAST_PAYLOAD_PATH)
    return {"ok": True, "last_payload": last}

@app.get("/signals/top3")
def signals_top3():
    last = _read_json(LAST_PAYLOAD_PATH)
    if not last:
        return {"ok": True, "top3": None}
    return {"ok": True, "top3": last.get("approved_top3") or last.get("top3") or last}

def _run_pipeline_internal() -> Dict[str, Any]:
    # Usa tu módulo ranking existente
    import ranking  # provided by your codebase
    payload = ranking.run_full_pipeline()
    return payload

def _run_and_prepare(token: str, send_tg: bool = True, dedupe_tg: bool = True) -> Dict[str, Any]:
    _auth_or_403(token)
    payload = _run_pipeline_internal()
    # Guardamos siempre el payload crudo
    base = {
        "as_of": payload.get("as_of"),
        "top50": payload.get("top50", []),
        "top3_factors": payload.get("top3_factors", []),
        "diag": payload.get("diag"),
    }

    # Intentamos reglas de pre-envío
    signals: List[Dict[str, Any]] = []
    used_presend = False
    try:
        presend_out = _use_presend_rules(base)
        # Esperamos una lista de señales aprobadas
        if isinstance(presend_out, list) and len(presend_out) > 0:
            signals = presend_out
            used_presend = True
    except Exception:
        used_presend = False

    # Si no hubo signals, caemos al fallback con top3 del ranking (o top50 primeros 3)
    if not signals:
        # Determinar top3 tickers
        top3_ticks: List[str] = []
        if isinstance(base.get("top3_factors"), list) and base["top3_factors"]:
            for it in base["top3_factors"]:
                if isinstance(it, dict) and it.get("ticker"):
                    top3_ticks.append(it["ticker"])
                elif isinstance(it, dict) and (it.get("symbol") or it.get("code")):
                    top3_ticks.append(it.get("symbol") or it.get("code"))
                elif isinstance(it, str):
                    top3_ticks.append(it)
        if not top3_ticks:
            for it in base.get("top50", [])[:3]:
                if isinstance(it, dict) and (it.get("symbol") or it.get("code")):
                    top3_ticks.append(it.get("symbol") or it.get("code"))
                elif isinstance(it, str):
                    top3_ticks.append(it)

        signals = _fallback_build_signals(top3_ticks)

    # Construimos mensajes (y dedupe)
    msgs = _build_messages_from_signals(signals)
    result = {
        "ok": True,
        "as_of": base.get("as_of"),
        "approved_top3": signals,
        "used_presend_rules": used_presend,
        "sent_tg": False,
        "tg_result": None,
    }

    # Persistir payload enriquecido
    _persist_last_payload(result)

    if send_tg:
        tg_res = _send_tg_messages(msgs, dedupe=dedupe_tg)
        result["sent_tg"] = tg_res.get("sent", False)
        result["tg_result"] = tg_res

    return result

@app.get("/signals/run-top3")
def run_top3(token: str, send_tg: bool = True, dedupe_tg: bool = True):
    return _run_and_prepare(token=token, send_tg=send_tg, dedupe_tg=dedupe_tg)

# Compat con tu ruta anterior
@app.get("/rank/run-top3")
def run_top3_compat(token: str, send_tg: bool = True, dedupe_tg: bool = True):
    return _run_and_prepare(token=token, send_tg=send_tg, dedupe_tg=dedupe_tg)
