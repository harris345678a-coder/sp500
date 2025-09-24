# -*- coding: utf-8 -*-
"""
App Evolutivo - Señales Top3 con enriquecimiento dinámico y envío a Telegram
----------------------------------------------------------------------------
- Endpoint principal: /signals/run-top3?token=123&send_tg=true&dedupe_tg=true
- Compatibilidad: /rank/run-top3 (misma lógica)
- Usa ranking.run_full_pipeline() para obtener top50 y top3_factors
- Si existe presend_rules.build_top3_signals(payload) lo usa; si no, aplica fallback estricto
- Enriquecimiento dinámico (ATR/ADX/EMA/MOM, TG/SL/TP1/TP2, RR) con datos en vivo (yfinance)
- Mensajería Telegram en HTML con escape seguro (sin 400 de "can't parse entities")
- Persistencia de último payload y digest para dedupe y auditoría

Requiere en el entorno (Render):
- API_TOKEN=123
- TELEGRAM_BOT_TOKEN ó BOT_TOKEN
- TELEGRAM_CHAT_ID ó CHAT_ID
- (opcional) TELEGRAM_NOTIFY_EMPTY=true

Coeficientes ATR (ajustables):
- LONG: SL = TG - 1.20*ATR, TP1 = TG + 1.00*ATR, TP2 = TG + 1.90*ATR
- SHORT: inverso
"""

import os
import io
import json
import math
import time
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------- Logging -------------------------------------
logger = logging.getLogger("app_evolutivo")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---------------------------- Config --------------------------------------
API_TOKEN = os.getenv("API_TOKEN", "123")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID")
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "1") not in ("0", "false", "False", "")

ADX_MIN = float(os.getenv("ADX_MIN", "15.0"))
DOLLAR_VOL_MIN = float(os.getenv("DOLLAR_VOL_MIN", "5000000"))
ATR_N = int(os.getenv("ATR_N", "14"))
EMA_FAST = int(os.getenv("EMA_FAST", "20"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "50"))
MOM_N = int(os.getenv("MOM_N", "63"))

STATE_DIR = "/tmp/evolutivo_state"
os.makedirs(STATE_DIR, exist_ok=True)
LAST_PAYLOAD_PATH = os.path.join(STATE_DIR, "last_payload.json")
LAST_DIGEST_PATH  = os.path.join(STATE_DIR, "last_top3_digest.txt")

# ---------------------------- Imports dinámicos ---------------------------
def _import_ranking():
    try:
        import ranking  # type: ignore
        return ranking
    except Exception as e:
        logger.exception("No se pudo importar 'ranking'")
        raise

def _import_presend_rules():
    try:
        import presend_rules  # type: ignore
        return presend_rules
    except Exception:
        return None

# ---------------------------- Utilidades ----------------------------------
def tg_escape(text: str) -> str:
    """Escapa HTML para Telegram parse_mode=HTML (permitimos poner <b> en plantillas)."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))

def _round(x: float, nd: int = 2) -> float:
    try:
        return round(float(x), nd)
    except Exception:
        return float("nan")

def _digest_for_signals(signals: List[Dict[str, Any]]) -> str:
    """Digest determinístico de señales finales para dedupe (ticker, side, TG, SL, TP1, TP2)."""
    key_items = []
    for s in sorted(signals, key=lambda z: (z.get("ticker",""), z.get("side",""))):
        key_items.append(f'{s.get("ticker")}:{s.get("side")}:{s.get("tg")}:{s.get("sl")}:{s.get("tp1")}:{s.get("tp2")}')
    raw = "|".join(key_items)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------- Indicadores ---------------------------------
import pandas as pd

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    # Wilder smoothing: ATR_t = (ATR_{t-1}*(n-1) + TR_t)/n
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    return atr

def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > -minus_dm), 0.0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > -plus_dm), 0.0)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/n, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/n, adjust=False).mean() / atr)
    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA) ) * 100
    adx = dx.ewm(alpha=1/n, adjust=False).mean()
    return adx

def _avg_dollar_vol(df: pd.DataFrame, n: int = 20) -> float:
    dv = (df["Close"] * df["Volume"]).tail(n).mean()
    return float(dv) if pd.notna(dv) else 0.0

# ---------------------------- Datos en vivo -------------------------------
def _get_live_df(ticker: str, period: str = "2y", interval: str = "1d") -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:
        logger.error("yfinance no disponible: %s", e)
        return None
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False, threads=False)
        if df is None or df.empty:
            logger.warning("Sin datos para %s", ticker)
            return None
        df = df.dropna()
        return df
    except Exception as e:
        logger.error("Error yfinance para %s: %s", ticker, e)
        return None

def _get_last_price(ticker: str, df: Optional[pd.DataFrame]) -> Optional[float]:
    px = None
    if df is not None and not df.empty:
        try:
            px = float(df["Close"].iloc[-1])
        except Exception:
            px = None
    if px is None:
        try:
            import yfinance as yf  # type: ignore
            t = yf.Ticker(ticker)
            fi = getattr(t, "fast_info", None)
            if fi and "last_price" in fi:
                px = float(fi["last_price"])
        except Exception:
            pass
    return px

# ---------------------------- Fallback estricto ---------------------------
def _analyze_one_ticker(ticker: str) -> Optional[Dict[str, Any]]:
    df = _get_live_df(ticker, period="2y", interval="1d")
    if df is None or df.empty or df.shape[0] < max(MOM_N+1, EMA_SLOW+1, ATR_N+1):
        return {"ticker": ticker, "reason_excluded": "no_data"}

    close = df["Close"]
    ema_fast = _ema(close, EMA_FAST)
    ema_slow = _ema(close, EMA_SLOW)
    mom = close - close.shift(MOM_N)
    atr = _atr(df, ATR_N)
    adx = _adx(df, ATR_N)
    dv = _avg_dollar_vol(df, n=20)

    ema_ok = bool(ema_fast.iloc[-1] > ema_slow.iloc[-1])
    mom_ok = bool(mom.iloc[-1] > 0)
    adx_ok = bool(adx.iloc[-1] >= ADX_MIN)
    liquid_ok = bool(dv >= DOLLAR_VOL_MIN)

    side = None
    if ema_ok and mom_ok:
        side = "LONG"
    elif (not ema_ok) and (not mom_ok):
        side = "SHORT"
    else:
        # Condiciones mixtas => no señal clara
        return {"ticker": ticker, "reason_excluded": "mixed_conditions"}

    if not adx_ok:
        return {"ticker": ticker, "reason_excluded": "adx_low"}
    if not liquid_ok:
        return {"ticker": ticker, "reason_excluded": "illiquid"}

    tg = _get_last_price(ticker, df)
    if tg is None:
        return {"ticker": ticker, "reason_excluded": "no_price"}
    atr_val = float(atr.iloc[-1])

    # Niveles por ATR (ajustables)
    if side == "LONG":
        sl = tg - 1.20*atr_val
        tp1 = tg + 1.00*atr_val
        tp2 = tg + 1.90*atr_val
        rr = (tp2 - tg) / max(tg - sl, 1e-9)
    else:
        sl = tg + 1.20*atr_val
        tp1 = tg - 1.00*atr_val
        tp2 = tg - 1.90*atr_val
        rr = (tg - tp2) / max(sl - tg, 1e-9)

    return {
        "ticker": ticker,
        "side": side,
        "tg": round(tg, 2),
        "sl": round(sl, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
        "atr": round(atr_val, 2),
        "rr": round(rr, 2),
        "strategy": "EMA20/50 + MOM63 + ADX + ATR",
    }

def _fallback_build_signals(top3_tickers: List[str]) -> List[Dict[str, Any]]:
    final = []
    excluded = []
    for tk in top3_tickers:
        res = _analyze_one_ticker(tk)
        if not res:
            continue
        if "reason_excluded" in res:
            excluded.append([tk, res["reason_excluded"]])
            continue
        final.append(res)
    return final

# ---------------------------- Enriquecimiento -----------------------------
def _enrich_targets(sig: Dict[str, Any]) -> Dict[str, Any]:
    """Completa TG/SL/TP1/TP2/ATR/RR con ATR si faltan, usando datos en vivo."""
    ticker = sig.get("ticker")
    side = sig.get("side")
    if not ticker:
        return sig
    df = _get_live_df(ticker, period="2y", interval="1d")
    if df is None or df.empty:
        return sig

    atr_val = float(_atr(df, ATR_N).iloc[-1])
    tg = sig.get("tg") or _get_last_price(ticker, df)
    if tg is None:
        return sig

    need_levels = any(sig.get(k) is None for k in ("sl", "tp1", "tp2"))
    if need_levels and side:
        if side.upper() == "LONG":
            sl = tg - 1.20*atr_val
            tp1 = tg + 1.00*atr_val
            tp2 = tg + 1.90*atr_val
        else:
            sl = tg + 1.20*atr_val
            tp1 = tg - 1.00*atr_val
            tp2 = tg - 1.90*atr_val
        sig["sl"] = round(sl, 2)
        sig["tp1"] = round(tp1, 2)
        sig["tp2"] = round(tp2, 2)

    # ATR y RR
    sig["atr"] = round(atr_val, 2)
    if "rr" not in sig or sig.get("rr") is None:
        if side and all(k in sig for k in ("sl","tp2","tg")):
            tg = float(sig["tg"])
            sl = float(sig["sl"])
            tp2 = float(sig["tp2"])
            if side.upper() == "LONG":
                rr = (tp2 - tg) / max(tg - sl, 1e-9)
            else:
                rr = (tg - tp2) / max(sl - tg, 1e-9)
            sig["rr"] = round(rr, 2)

    return sig

# ---------------------------- Telegram ------------------------------------
import requests

def _telegram_enabled() -> (bool, str):
    if not TELEGRAM_ENABLED:
        return False, "disabled_by_env"
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False, "missing_env(TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID)"
    return True, "ok"

def _format_signal_html(sig: Dict[str, Any], as_of: str) -> str:
    t = tg_escape(str(sig.get("ticker","-")))
    side = tg_escape(str(sig.get("side","-")))
    strat = tg_escape(str(sig.get("strategy","")))
    tg_px = sig.get("tg"); sl = sig.get("sl"); tp1 = sig.get("tp1"); tp2 = sig.get("tp2")
    atr = sig.get("atr"); rr = sig.get("rr")

    lines = []
    # Encabezado
    emoji = "📈" if side.upper() == "LONG" else "📉"
    lines.append(f"{emoji} <b>{t}</b> • {side}")
    if strat:
        lines.append(f"📌 Estrategia: {strat}")
    # Niveles
    def fnum(x): 
        try: 
            return f"{float(x):.2f}"
        except Exception:
            return "-"
    lines.append(f"🎯 TG {fnum(tg_px)}  |  ⛔ SL {fnum(sl)}")
    lines.append(f"🥇 TP1 {fnum(tp1)}  |  🥈 TP2 {fnum(tp2)}")
    # Métricas
    add = []
    if atr is not None: add.append(f"ATR14 {fnum(atr)}")
    if rr is not None: add.append(f"RR(TP2) {fnum(rr)}")
    add.append(f"as_of {tg_escape(as_of)}")
    lines.append(" · ".join(add))
    return "\n".join(lines)

def _send_telegram_messages(signals: List[Dict[str, Any]], as_of: str, dedupe_ok: bool) -> Dict[str, Any]:
    ok, reason = _telegram_enabled()
    info = {"attempted": True, "enabled": ok, "tickers": [s.get("ticker") for s in signals]}
    if not ok:
        info["reason"] = reason
        return {"notified": False, "notify_info": info}

    # Dedupe por digest
    digest = _digest_for_signals(signals)
    info["digest"] = digest
    if dedupe_ok:
        try:
            if os.path.exists(LAST_DIGEST_PATH):
                last = open(LAST_DIGEST_PATH, "r").read().strip()
                if last == digest:
                    info["reason"] = "duplicate_digest"
                    return {"notified": False, "notify_info": info}
        except Exception:
            pass

    sent_all = True
    status = 200
    try:
        for sig in signals:
            msg = _format_signal_html(sig, as_of)
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True}
            r = requests.post(url, json=payload, timeout=15)
            if r.status_code != 200:
                sent_all = False
                status = r.status_code
                info["reason"] = f"send_failed(status={r.status_code} body={r.text})"
                logger.error("Telegram send failed: %s %s", r.status_code, r.text)
        if sent_all:
            with open(LAST_DIGEST_PATH, "w") as f:
                f.write(digest)
    except Exception as e:
        sent_all = False
        status = 500
        info["reason"] = f"send_exception({e})"
        logger.exception("Telegram send exception")

    info["status"] = status
    info["response"] = "ok" if sent_all else info.get("reason","send_partial")
    return {"notified": sent_all, "notify_info": info}

# ---------------------------- Persistencia --------------------------------
def _persist_payload(payload: Dict[str, Any]) -> None:
    try:
        with open(LAST_PAYLOAD_PATH, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("No se pudo persistir payload")

def _load_last_payload() -> Optional[Dict[str, Any]]:
    try:
        if os.path.exists(LAST_PAYLOAD_PATH):
            with open(LAST_PAYLOAD_PATH, "r") as f:
                return json.load(f)
    except Exception:
        logger.exception("No se pudo leer payload previo")
    return None

# ---------------------------- FastAPI -------------------------------------
app = FastAPI(title="App Evolutivo", version="1.0.0")

def _ok_token(token: str) -> bool:
    return token == API_TOKEN

def _call_run_full_pipeline() -> Dict[str, Any]:
    ranking = _import_ranking()
    # Algunos módulos no aceptan kwargs 'audit'; probamos seguro
    try:
        # Intento con audit, si falla por TypeError, reintento sin audit
        return ranking.run_full_pipeline(audit=False)  # type: ignore
    except TypeError:
        return ranking.run_full_pipeline()  # type: ignore

def _use_presend_rules(payload: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    mod = _import_presend_rules()
    if not mod:
        return None
    try:
        if hasattr(mod, "build_top3_signals"):
            return mod.build_top3_signals(payload)  # type: ignore
    except Exception:
        logger.exception("Error en presend_rules.build_top3_signals")
    return None

@app.get("/healthz")
def healthz():
    ok, reason = _telegram_enabled()
    last_digest = None
    try:
        if os.path.exists(LAST_DIGEST_PATH):
            last_digest = open(LAST_DIGEST_PATH).read().strip()
    except Exception:
        pass
    return {
        "ok": True,
        "time": datetime.now(timezone.utc).isoformat(),
        "telegram_enabled": ok,
        "telegram_reason": reason if not ok else "ok",
        "has_last_payload": os.path.exists(LAST_PAYLOAD_PATH),
        "last_digest": last_digest,
        "version": "1.0.0"
    }

@app.get("/signals/status")
def signals_status():
    p = _load_last_payload()
    if not p:
        return {"ok": True, "has_payload": False}
    return {
        "ok": True,
        "has_payload": True,
        "as_of": p.get("as_of"),
        "top3_tickers": [s.get("ticker") for s in p.get("final_signals", [])],
        "digest": _digest_for_signals(p.get("final_signals", [])),
    }

@app.get("/signals/top3")
def signals_top3():
    p = _load_last_payload()
    if not p:
        return {"ok": True, "has_payload": False}
    return p

@app.get("/signals/run-top3")
def run_top3(
    token: str = Query(...),
    send_tg: bool = Query(True),
    dedupe_tg: bool = Query(True),
):
    if not _ok_token(token):
        raise HTTPException(status_code=403, detail="bad token")

    t0 = time.time()
    # 1) Ejecutar ranking
    payload = _call_run_full_pipeline()
    as_of = payload.get("as_of") or datetime.utcnow().date().isoformat()

    # 2) Intentar construir señales con presend_rules
    final_signals: Optional[List[Dict[str, Any]]] = None
    try:
        final_signals = _use_presend_rules(payload)
    except Exception:
        final_signals = None

    # 3) Fallback estricto si no hay señales válidas
    diag = payload.get("diag", {}).copy() if isinstance(payload.get("diag"), dict) else {}
    if not final_signals:
        top3_factors = payload.get("top3_factors") or []
        top3_tickers = [x.get("ticker") if isinstance(x, dict) else x for x in top3_factors]
        top3_tickers = [t for t in top3_tickers if t] or (payload.get("top50", [])[:3])
        final_signals = _fallback_build_signals(top3_tickers)
        diag["fallback_used"] = True
    else:
        diag["fallback_used"] = False

    # 4) Enriquecer niveles/ATR/RR de forma robusta
    enriched_signals = []
    for s in final_signals:
        enriched_signals.append(_enrich_targets(s.copy()))
    final_signals = enriched_signals

    # 5) Persistencia para auditoría y dedupe
    out = {
        "ok": True,
        "took_s": round(time.time() - t0, 2),
        "as_of": as_of,
        "top50": payload.get("top50", [])[:50],
        "top3_factors": payload.get("top3_factors", [])[:3],
        "diag": diag,
        "final_signals": final_signals,
        "notified": False,
    }

    # 6) Envío a Telegram (solo approved_top3 / final_signals)
    notify_info = {"attempted": False}
    if send_tg and final_signals:
        ret = _send_telegram_messages(final_signals, as_of, dedupe_ok=dedupe_tg)
        out["notified"] = ret.get("notified", False)
        notify_info = ret.get("notify_info", notify_info)
    else:
        notify_info["reason"] = "send_tg_false_or_empty_signals"
    out["notify_info"] = notify_info

    _persist_payload(out)
    logger.info("Pipeline listo · as_of=%s · top50=%s · top3_factors=%s · %.2fs",
                as_of, len(out["top50"]), len(out["top3_factors"]), out["took_s"])
    return JSONResponse(out)

# Compat anterior
@app.get("/rank/run-top3")
def run_top3_compat(token: str = Query(...), send_tg: bool = Query(True), dedupe_tg: bool = Query(True)):
    # Reutiliza la misma lógica
    return run_top3(token=token, send_tg=send_tg, dedupe_tg=dedupe_tg)
