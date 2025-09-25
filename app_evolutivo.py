\
# app_evolutivo.py
# ------------------------------------------------------------
# Servicio FastAPI para ejecutar el pipeline, armar señales top3
# (largos/cortos), enriquecer niveles (TG/SL/TP1/TP2) con ATR,
# y enviar a Telegram en formato HTML con dedupe/firma y persistencia.
#
# NOTA: Este archivo NO toca tu lógica de ranking ni tus presend_rules.
#       Sólo adapta el payload si viene en formato "lista de strings"
#       para evitar el error AttributeError: 'str' object has no attribute 'get'.
#       Si presend_rules falla, usa un fallback estable.
# ------------------------------------------------------------
import os
import io
import json
import time
import math
import copy
import logging
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from fastapi import FastAPI, Query, HTTPException, Response
from fastapi.responses import JSONResponse

# Módulos propios (debes tenerlos en el proyecto)
import ranking  # debe exponer run_full_pipeline()
try:
    import presend_rules  # debe exponer build_top3_signals(payload)
    _PRESEND_AVAILABLE = True
except Exception:
    presend_rules = None  # type: ignore
    _PRESEND_AVAILABLE = False

# -----------------------------
# Config & Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("app_evolutivo")

API_TOKEN = os.getenv("API_TOKEN", "123")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", os.getenv("BOT_TOKEN", ""))
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", os.getenv("CHAT_ID", ""))
NOTIFY_EMPTY = os.getenv("TELEGRAM_NOTIFY_EMPTY", "false").lower() == "true"

# Umbrales (puedes ajustar por env)
ADX_MIN = float(os.getenv("ADX_MIN", "15"))
DOLLAR_VOL_MIN = float(os.getenv("DOLLAR_VOL_MIN", "5000000"))
ATR_N = int(os.getenv("ATR_N", "14"))
EMA_FAST = int(os.getenv("EMA_FAST", "20"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "50"))
MOM_N = int(os.getenv("MOM_N", "63"))

# Multiplicadores de ATR (puedes cambiar por env si necesitas)
SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.2"))
TP1_ATR_MULT = float(os.getenv("TP1_ATR_MULT", "1.0"))
TP2_ATR_MULT = float(os.getenv("TP2_ATR_MULT", "1.9"))

STATE_DIR = "/tmp/evolutivo_state"
os.makedirs(STATE_DIR, exist_ok=True)
STATE_PAYLOAD = os.path.join(STATE_DIR, "last_payload.json")
STATE_DIGEST = os.path.join(STATE_DIR, "last_digest.txt")

app = FastAPI(title="Evolutivo Signals Service", version="1.0.0")


# -----------------------------
# Utilidades de series
# -----------------------------
def _ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()

def _momentum(series: pd.Series, n: int) -> pd.Series:
    return series / series.shift(n) - 1.0

def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = _true_range(df)
    return tr.rolling(n).mean()

def _avg_dollar_vol(df: pd.DataFrame, n: int = 20) -> float:
    # Robusto: soporta futuros sin 'Volume' (devuelve 0), y siempre retorna escalar
    if "Close" not in df.columns:
        return 0.0
    vol = df["Volume"] if "Volume" in df.columns else pd.Series(0.0, index=df.index)
    dv = (df["Close"] * vol).rolling(n).mean()
    if len(dv) == 0:
        return 0.0
    last = dv.iloc[-1]
    return float(last) if pd.notna(last) else 0.0

def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    # Implementación clásica de ADX
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0.0
    minus_dm[minus_dm < 0] = 0.0

    tr = _true_range(df)
    atr = tr.rolling(n).mean()

    plus_di = 100 * (plus_dm.rolling(n).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(n).mean() / atr)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    adx = dx.rolling(n).mean()
    return adx


# -----------------------------
# Telegram helpers
# -----------------------------
_HTML_REPLACE = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
}

def _html_escape(s: str) -> str:
    # Escapa &, <, > (no incluimos comillas para evitar problemas con parse_mode)
    return s.translate(str.maketrans(_HTML_REPLACE))

def _send_telegram(text: str) -> Tuple[bool, str]:
    token = BOT_TOKEN
    chat_id = CHAT_ID
    if not token or not chat_id:
        return False, "missing_env(TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID)"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.ok:
            return True, "ok"
        return False, f"send_failed(status={r.status_code} body={r.text})"
    except Exception as e:
        return False, f"send_exception({type(e).__name__}: {e})"


# -----------------------------
# Build de señales: enriquecimiento ATR
# -----------------------------
def _enrich_targets(sig: Dict[str, Any]) -> Dict[str, Any]:
    # Si ya tiene niveles, se respetan. Si faltan, se calculan con ATR.
    # Requiere: sig['side'] in {'LONG','SHORT'}, sig['tg'], sig['atr']
    side = sig.get("side")
    tg = float(sig.get("tg", 0.0))
    atr = float(sig.get("atr", 0.0))

    sl = sig.get("sl")
    tp1 = sig.get("tp1")
    tp2 = sig.get("tp2")

    if atr <= 0:
        # Si no hay ATR, deja niveles tal cual estén.
        pass
    else:
        if side == "LONG":
            sl = float(sl) if sl is not None else tg - SL_ATR_MULT * atr
            tp1 = float(tp1) if tp1 is not None else tg + TP1_ATR_MULT * atr
            tp2 = float(tp2) if tp2 is not None else tg + TP2_ATR_MULT * atr
        elif side == "SHORT":
            sl = float(sl) if sl is not None else tg + SL_ATR_MULT * atr
            tp1 = float(tp1) if tp1 is not None else tg - TP1_ATR_MULT * atr
            tp2 = float(tp2) if tp2 is not None else tg - TP2_ATR_MULT * atr

    sig["sl"] = float(sl) if sl is not None else None
    sig["tp1"] = float(tp1) if tp1 is not None else None
    sig["tp2"] = float(tp2) if tp2 is not None else None

    # RR contra TP2 si falta
    rr = sig.get("rr")
    if rr is None and sig.get("sl") is not None and sig.get("tp2") is not None:
        if side == "LONG":
            risk = max(tg - sig["sl"], 1e-9)
            reward = sig["tp2"] - tg
        else:
            risk = max(sig["sl"] - tg, 1e-9)
            reward = tg - sig["tp2"]
        rr = reward / risk if risk > 0 else None
        sig["rr"] = float(rr) if rr is not None else None

    return sig


# -----------------------------
# Fallback builder (si presend_rules falla)
# -----------------------------
def _fetch_df(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    df = df.dropna().copy()
    if "Adj Close" in df.columns:
        df.rename(columns={"Adj Close": "AdjClose"}, inplace=True)
    return df

def _analyze_one_ticker(ticker: str) -> Optional[Dict[str, Any]]:
    df = _fetch_df(ticker)
    if df.empty or "Close" not in df.columns:
        return None

    close = df["Close"]
    ema_fast = _ema(close, EMA_FAST)
    ema_slow = _ema(close, EMA_SLOW)
    mom = _momentum(close, MOM_N)
    adx = _adx(df, n=ATR_N)
    atr = _atr(df, n=ATR_N)

    last = df.index[-1]
    last_price = float(close.iloc[-1])
    last_ema_fast = float(ema_fast.iloc[-1])
    last_ema_slow = float(ema_slow.iloc[-1])
    last_mom = float(mom.iloc[-1]) if not math.isnan(mom.iloc[-1]) else 0.0
    last_adx = float(adx.iloc[-1]) if not math.isnan(adx.iloc[-1]) else 0.0
    last_atr = float(atr.iloc[-1]) if not math.isnan(atr.iloc[-1]) else 0.0

    # Liquidez (dólar-vol 20)
    dv = _avg_dollar_vol(df, n=20)

    side = None
    reasons = []
    if last_adx >= ADX_MIN and last_ema_fast > last_ema_slow and last_mom > 0:
        side = "LONG"
        reasons = ["ema20>ema50", "momentum63_pos", f"adx>={ADX_MIN}"]
    elif last_adx >= ADX_MIN and last_ema_fast < last_ema_slow and last_mom < 0:
        side = "SHORT"
        reasons = ["ema20<ema50", "momentum63_neg", f"adx>={ADX_MIN}"]
    else:
        # No cumple filtros básicos
        return None

    sig = {
        "symbol": ticker,
        "code": ticker,
        "side": side,
        "strategy": "EMA/MOM/ADX",
        "tg": last_price,
        "atr": last_atr,
        "avg_dollar_vol20": dv,
        "reasons": reasons,
        "as_of": str(last.date()),
    }
    sig = _enrich_targets(sig)
    return sig

def _fallback_build_signals(top3_tickers: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for tk in top3_tickers:
        try:
            res = _analyze_one_ticker(tk)
            if res is not None:
                out.append(res)
        except Exception as e:
            log.exception("fallback analyze failed for %s: %s", tk, e)
    return out


# -----------------------------
# Adaptador para presend_rules
# -----------------------------
def _adapt_payload_for_presend(payload: Dict[str, Any]) -> Dict[str, Any]:
    p = copy.deepcopy(payload)

    # top50: si viene como lista de strings, conviértelo a lista de dicts
    t50 = p.get("top50", [])
    if isinstance(t50, list) and len(t50) > 0 and isinstance(t50[0], str):
        p["top50"] = [{"symbol": s, "code": s} for s in t50]

    # top3_factors: asegurar dict con 'ticker'/'symbol' y 'reasons'
    t3f = p.get("top3_factors", [])
    fixed = []
    for itm in t3f:
        if isinstance(itm, str):
            fixed.append({"ticker": itm, "reasons": []})
        elif isinstance(itm, dict):
            # normalizar claves
            tk = itm.get("ticker") or itm.get("symbol") or itm.get("code")
            reasons = itm.get("reasons") or []
            fixed.append({"ticker": tk, "reasons": reasons})
        else:
            # desconocido -> ignora
            pass
    p["top3_factors"] = fixed

    # factor genérico solicitado por algunos presend_rules
    if "factors" not in p and "top50" in p:
        p["factors"] = p["top50"]

    return p


def _use_presend_rules(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not _PRESEND_AVAILABLE or presend_rules is None:
        raise RuntimeError("presend_rules_not_available")
    try:
        p = _adapt_payload_for_presend(payload)
        out = presend_rules.build_top3_signals(p)  # type: ignore
        if not isinstance(out, list):
            raise ValueError("presend_rules.build_top3_signals must return list")
        # Enriquecer niveles si faltan TG/SL/TP1/TP2 usando ATR live
        enriched = []
        for s in out:
            try:
                sym = s.get("symbol") or s.get("code") or s.get("ticker")
                if not sym:
                    continue
                # ATR live
                df = _fetch_df(sym)
                if df.empty or "Close" not in df.columns:
                    atr = 0.0
                    tg = float(s.get("tg", 0.0))
                else:
                    atr = float(_atr(df, ATR_N).iloc[-1])
                    tg = float(s.get("tg", float(df["Close"].iloc[-1])))
                s["tg"] = tg
                s["atr"] = float(s.get("atr", atr))
                # normalizar side
                side_raw = (s.get("side") or "").upper()
                if side_raw not in ("LONG", "SHORT"):
                    # heurística en caso de que presend_rules omita side
                    if s.get("ema20_gt_ema50") or s.get("bias") == "long":
                        side_raw = "LONG"
                    elif s.get("ema20_lt_ema50") or s.get("bias") == "short":
                        side_raw = "SHORT"
                s["side"] = side_raw if side_raw in ("LONG", "SHORT") else "LONG"
                s = _enrich_targets(s)
                enriched.append(s)
            except Exception:
                logging.exception("enrich failed for %s", s)
        return enriched
    except Exception as e:
        log.exception("Error en presend_rules.build_top3_signals")
        raise


# -----------------------------
# Persistencia & Dedupe
# -----------------------------
def _save_payload(payload: Dict[str, Any]) -> None:
    try:
        with open(STATE_PAYLOAD, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        log.exception("no pudo persistir STATE_PAYLOAD")

def _load_payload() -> Optional[Dict[str, Any]]:
    try:
        if os.path.exists(STATE_PAYLOAD):
            with open(STATE_PAYLOAD, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        log.exception("no pudo cargar STATE_PAYLOAD")
    return None

def _read_digest() -> str:
    try:
        if os.path.exists(STATE_DIGEST):
            return open(STATE_DIGEST, "r", encoding="utf-8").read().strip()
    except Exception:
        log.exception("no pudo leer STATE_DIGEST")
    return ""

def _write_digest(digest: str) -> None:
    try:
        with open(STATE_DIGEST, "w", encoding="utf-8") as f:
            f.write(digest)
    except Exception:
        log.exception("no pudo escribir STATE_DIGEST")


# -----------------------------
# Formateo de mensajes
# -----------------------------
def _fmt_num(x: Optional[float]) -> str:
    if x is None:
        return "-"
    try:
        if x >= 1000 or x <= -1000:
            return f"{x:,.2f}"
        return f"{x:.4f}"
    except Exception:
        return str(x)

def _format_signal_html(sig: Dict[str, Any]) -> str:
    sym = _html_escape(str(sig.get("symbol") or sig.get("code") or sig.get("ticker") or ""))
    strat = _html_escape(str(sig.get("strategy", "")))
    side = _html_escape(str(sig.get("side", "")))
    tg = _fmt_num(sig.get("tg"))
    sl = _fmt_num(sig.get("sl"))
    tp1 = _fmt_num(sig.get("tp1"))
    tp2 = _fmt_num(sig.get("tp2"))
    rr = sig.get("rr")
    rr_txt = "-" if rr is None else f"{float(rr):.2f}"
    atr = sig.get("atr")
    atr_txt = "-" if atr is None else _fmt_num(float(atr))

    line1 = f"<b>{sym} • {side} • {strat}</b>"
    line2 = f"TG: {tg}  |  SL: {sl}  |  TP1: {tp1}  |  TP2: {tp2}"
    line3 = f"RR: {rr_txt}  |  ATR: {atr_txt}"
    return "\n".join([line1, line2, line3])


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "telegram_enabled": bool(BOT_TOKEN and CHAT_ID),
        "modules": {
            "ranking": True,
            "presend_rules": _PRESEND_AVAILABLE,
        },
    }

@app.get("/signals/status")
def signals_status():
    payload = _load_payload()
    digest = _read_digest()
    return {
        "ok": True,
        "has_payload": payload is not None,
        "digest": digest,
        "payload_as_of": payload.get("as_of") if payload else None,
        "telegram_enabled": bool(BOT_TOKEN and CHAT_ID),
    }

@app.get("/signals/top3")
def get_last_payload():
    payload = _load_payload()
    if not payload:
        raise HTTPException(404, "No hay payload persistido")
    return payload

def _auth_or_403(token: str):
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="token inválido")

def _digest_messages(msgs: List[str]) -> str:
    joined = "\n\n".join(msgs).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()

def _build_messages_for_tg(signals: List[Dict[str, Any]]) -> List[str]:
    return [_format_signal_html(s) for s in signals]

def _extract_top3_tickers(payload: Dict[str, Any]) -> List[str]:
    top3 = []
    t3f = payload.get("top3_factors", [])
    for itm in t3f:
        if isinstance(itm, dict):
            tk = itm.get("ticker") or itm.get("symbol") or itm.get("code")
            if tk:
                top3.append(tk)
        elif isinstance(itm, str):
            top3.append(itm)
    if not top3:
        # fallback: primeros 3 del top50
        t50 = payload.get("top50", [])
        if isinstance(t50, list):
            for itm in t50[:3]:
                if isinstance(itm, str):
                    top3.append(itm)
                elif isinstance(itm, dict):
                    tk = itm.get("symbol") or itm.get("code") or itm.get("ticker")
                    if tk:
                        top3.append(tk)
    # quitar duplicados preservando orden
    seen = set()
    unique = []
    for t in top3:
        if t not in seen:
            unique.append(t)
            seen.add(t)
    return unique[:3]


@app.get("/signals/run-top3")
def run_top3(
    token: str = Query(...),
    send_tg: bool = Query(True),
    dedupe_tg: bool = Query(True)
):
    _auth_or_403(token)
    t0 = time.time()

    # 1) Ranking pipeline original
    payload = ranking.run_full_pipeline()
    # payload esperado: dict con keys como 'as_of', 'top50', 'top3_factors', etc.
    as_of = payload.get("as_of") or datetime.date.today().isoformat()

    # 2) Intentar reglas de pre-envío
    notified = False
    notify_info: Dict[str, Any] = {}
    try:
        signals = _use_presend_rules(payload)
    except Exception:
        # Fallback estable
        top3_tickers = _extract_top3_tickers(payload)
        signals = _fallback_build_signals(top3_tickers)

    # 3) Filtrado final: quitar duplicados por símbolo-lado y asegurar 3 máx
    final: List[Dict[str, Any]] = []
    seen_pairs = set()
    for s in signals:
        sym = s.get("symbol") or s.get("code") or s.get("ticker")
        side = (s.get("side") or "").upper()
        if not sym or side not in ("LONG", "SHORT"):
            continue
        key = (sym, side)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        final.append(s)
        if len(final) == 3:
            break

    # 4) Persistir
    out_payload = {
        "ok": True,
        "as_of": as_of,
        "approved_top3": final,
        "diag": {
            "universe_count": payload.get("diag", {}).get("universe_count"),
            "fetched_count": payload.get("diag", {}).get("fetched_count"),
            "excluded_count": payload.get("diag", {}).get("excluded_count"),
        },
    }
    _save_payload(out_payload)

    # 5) Telegram
    if send_tg:
        if not final and NOTIFY_EMPTY:
            msgs = ["<b>Sin señales aprobadas</b>"]
        else:
            msgs = _build_messages_for_tg(final)

        digest = _digest_messages(msgs)
        prev = _read_digest()
        if (not dedupe_tg) or (digest != prev):
            if not msgs:
                notified = False
                notify_info = {"attempted": False, "reason": "no_messages_to_send"}
            else:
                ok_all = True
                last_err = "ok"
                for m in msgs:
                    ok, info = _send_telegram(m)
                    if not ok:
                        ok_all = False
                        last_err = info
                        break
                notified = ok_all
                notify_info = {
                    "attempted": True,
                    "enabled": bool(BOT_TOKEN and CHAT_ID),
                    "status": 200 if ok_all else 500,
                    "response": "ok" if ok_all else last_err,
                }
                if ok_all:
                    _write_digest(digest)
        else:
            notified = False
            notify_info = {"attempted": True, "dedupe": "skipped_same_digest"}

    t1 = time.time()
    log.info("app_evolutivo | %s", json.dumps({"msg": f"Pipeline listo · as_of={as_of} · top3={len(final)} · {t1-t0:.2f}s"}))

    return {
        **out_payload,
        "notified": notified,
        "notify_info": notify_info,
        "took_s": round(t1 - t0, 2),
    }

# Compat antiguo
@app.get("/rank/run-top3")
def run_top3_compat(token: str = Query(...), send_tg: bool = Query(True), dedupe_tg: bool = Query(True)):
    return run_top3(token=token, send_tg=send_tg, dedupe_tg=dedupe_tg)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
