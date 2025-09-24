
# -*- coding: utf-8 -*-
"""
ranking.py — robust market scanner with strict filters and dynamic levels.
- Universe: ETFs/sectors/megacaps + GC=F/CL=F (no crypto). Can override with env UNIVERSE_TICKERS.
- Data: yfinance 2y/1D; indicators: EMA20/EMA50, Momentum63, ATR14 (Wilder), ADX14, DollarVol20.
- Filters:
  * Liquidity (avg dollar vol 20) except for whitelisted futures (GC=F, CL=F)
  * Trend & momentum (EMA20 vs EMA50, Momentum63)
  * Strength (ADX >= MIN_ADX)
  * Shorts supported simétricamente
- Top50: relleno con "relajados" solo para completar 50 (mantienen tendencia y liquidez). Top3 siempre estricto.
- Levels (dinámicos) con precio vivo si disponible: TG, SL, TP1, TP2 y TL (trailing start).
- Telegram: texto plano sin parse_mode; dedupe por hash diario en /tmp para evitar spam.
- API: run_full_pipeline(audit: bool=False, **_ignored) -> dict
"""

from __future__ import annotations
import os, json, time, math, hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
import logging

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

# ---------------- Logging -----------------
LOG_NAME = "ranking"
logger = logging.getLogger(LOG_NAME)
if not logger.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
logger.setLevel(logging.INFO)

def _log(tag: str, payload: Dict[str, Any]):
    logger.info("%s | %s", tag, json.dumps(payload, ensure_ascii=False))

# ---------------- Config ------------------
MIN_ADX = float(os.getenv("MIN_ADX", "15"))
MIN_DOLLAR_VOL20 = float(os.getenv("MIN_DOLLAR_VOL20", "5000000"))  # $5M
BREAKOUT_LOOKBACK = int(os.getenv("BREAKOUT_LOOKBACK", "55"))

K_SL_ATR = float(os.getenv("K_SL_ATR", "1.0"))
K_TP1_ATR = float(os.getenv("K_TP1_ATR", "1.0"))
K_TP2_ATR = float(os.getenv("K_TP2_ATR", "2.0"))
K_TL_ATR  = float(os.getenv("K_TL_ATR",  "1.0"))

ENABLE_TELEGRAM = os.getenv("ENABLE_TELEGRAM", "true").lower() != "false"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

FUTURES_LIQ_WHITELIST = {"GC=F", "CL=F"}

# ---------------- Universe ----------------
def default_universe() -> List[str]:
    """Universe amplio: ETFs core, sectores, temas, bonos, commodities, megacaps, semis, etc., + GC=F/CL=F."""
    etfs_core = [
        "SPY","QQQ","IWM","DIA","VTI","VOO","IVV","VTV","VOE","VUG","VGT","VHT","VFH",
        "VNQ","XLC","XLK","XLY","XLP","XLV","XLI","XLB","XLRE","XLU","XLF","XLE",
        "SOXX","SMH","XME","GDX","GDXJ","IBB","XBI","IYR","IYT","XRT","XAR","XTL",
        "GLD","SLV","DBC","DBA","USO","UNG","TLT","IEF","SHY","LQD","HYG","URA","TAN","OIH","XHB","ITB"
    ]
    megacaps = [
        "AAPL","MSFT","NVDA","GOOGL","GOOG","AMZN","META","TSLA","AVGO","ADBE","CSCO","CRM",
        "NFLX","AMD","INTC","QCOM","TXN","MU","AMAT","ASML",
        "JPM","BAC","WFC","GS","MS","BLK","C","V","MA","AXP","INTU",
        "XOM","CVX","COP","SLB","EOG","PSX",
        "UNH","JNJ","LLY","ABBV","MRK","PFE","TMO","DHR",
        "HD","LOW","COST","WMT","TGT","NKE","SBUX","MCD","BKNG",
        "CAT","DE","BA","GE","HON","UPS","FDX","MMM","ORCL","SAP",
        "PEP","KO","PG","CL","KHC","MDLZ","DIS","CMCSA","T","VZ"
    ]
    growth_mid = [
        "NOW","PANW","SNOW","NET","ZS","DDOG","SHOP","SQ","PYPL","UBER","LYFT","WBD"
    ]
    futures = ["GC=F","CL=F"]
    # Quitar duplicados manteniendo orden
    seen = set()
    out = []
    for t in etfs_core + megacaps + growth_mid + futures:
        if t not in seen:
            out.append(t); seen.add(t)
    return out

def load_universe() -> List[str]:
    custom = os.getenv("UNIVERSE_TICKERS", "").strip()
    if custom:
        # soporta coma o espacios
        raw = [x.strip() for x in custom.replace(" ", ",").split(",") if x.strip()]
        out, seen = [], set()
        for t in raw:
            if t not in seen: out.append(t); seen.add(t)
        tickers = out
    else:
        tickers = default_universe()
    # Auditoría
    sample = tickers[:10]
    _log("DATA_AUDIT | universe", {"count": len(tickers), "tickers_sample": sample})
    return tickers

# ---------------- Indicators ---------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def wilder_rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev_c = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_c).abs()
    tr3 = (l - prev_c).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr14(df: pd.DataFrame) -> pd.Series:
    tr = true_range(df["High"], df["Low"], df["Close"])
    return wilder_rma(tr, 14)

def adx14(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(high, low, close)
    atr = wilder_rma(tr, 14)
    plus_di = 100 * wilder_rma(pd.Series(plus_dm, index=high.index), 14) / atr
    minus_di = 100 * wilder_rma(pd.Series(minus_dm, index=high.index), 14) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = wilder_rma(dx, 14)
    return adx

def momentum63(close: pd.Series) -> pd.Series:
    return close.pct_change(63)

def dollar_vol20(df: pd.DataFrame) -> pd.Series:
    vol = df["Volume"].fillna(0)
    return (df["Close"] * vol).rolling(20, min_periods=20).mean()

# -------------- Data fetch -----------------
def fetch_history(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    if yf is None:
        raise RuntimeError("yfinance no disponible")
    # Primer intento: descarga por lote
    data: Dict[str, pd.DataFrame] = {}
    try:
        batch = yf.download(
            tickers=" ".join(tickers),
            period="2y", interval="1d",
            group_by="ticker", auto_adjust=False, threads=True, progress=False
        )
        # yfinance retorna MultiIndex cuando múltiples tickers
        if isinstance(batch.columns, pd.MultiIndex):
            for t in tickers:
                if (t in batch.columns.get_level_values(0)) and ("Close" in batch[t]):
                    df = batch[t].dropna(how="all")
                    data[t] = df
        else:
            # Solo un ticker
            df = batch.dropna(how="all")
            if not df.empty:
                data[tickers[0]] = df
    except Exception as e:
        _log("DATA_FETCH | batch_error", {"error": str(e)})

    # Fallback por ticker si faltan
    missing = [t for t in tickers if t not in data]
    for t in missing:
        try:
            df = yf.download(tickers=t, period="2y", interval="1d", auto_adjust=False, progress=False)
            df = df.dropna(how="all")
            if not df.empty:
                data[t] = df
            else:
                _log("yfinance", {"ticker": t, "error": "no_data"})
        except Exception as e:
            _log("yfinance", {"ticker": t, "error": str(e)})
    return data

# -------------- Compute indicators ---------
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["EMA20"] = ema(out["Close"], 20)
    out["EMA50"] = ema(out["Close"], 50)
    out["ATR14"] = atr14(out)
    out["ADX14"] = adx14(out)
    out["MOM63"] = momentum63(out["Close"])
    out["DollarVol20"] = dollar_vol20(out)
    return out

def latest_metrics(ticker: str, df: pd.DataFrame) -> Dict[str, Any]:
    row = df.iloc[-1]
    # Señales long/short
    long_ok = (row["EMA20"] > row["EMA50"]) and (row["MOM63"] > 0) and (row["ADX14"] >= MIN_ADX)
    short_ok = (row["EMA20"] < row["EMA50"]) and (row["MOM63"] < 0) and (row["ADX14"] >= MIN_ADX)
    # Liquidez (whitelist para futuros)
    if ticker in FUTURES_LIQ_WHITELIST:
        liquid = True
    else:
        liquid = bool(row["DollarVol20"] >= MIN_DOLLAR_VOL20) if not math.isnan(row["DollarVol20"]) else False

    reason_excluded = None
    if df.shape[0] < 100:
        reason_excluded = "too_few_rows"
    elif not liquid:
        reason_excluded = "illiquid"
    elif (not long_ok) and (not short_ok):
        # Más granular
        if row["ADX14"] < MIN_ADX:
            reason_excluded = "adx_low"
        elif row["EMA20"] <= row["EMA50"] and row["MOM63"] > 0:
            reason_excluded = "ema20<=ema50"
        elif row["EMA20"] >= row["EMA50"] and row["MOM63"] < 0:
            reason_excluded = "momentum63<=0"
        else:
            reason_excluded = "no_side"

    # scoring (positivo para long, para short usamos abs momentum)
    mom = float(row["MOM63"]) if not math.isnan(row["MOM63"]) else 0.0
    adx = float(row["ADX14"]) if not math.isnan(row["ADX14"]) else 0.0
    score_long = adx * max(mom, 0)
    score_short = adx * max(-mom, 0)

    return {
        "ticker": ticker,
        "rows": int(df.shape[0]),
        "start": str(df.index.min().date()) if df.shape[0] else None,
        "end": str(df.index.max().date()) if df.shape[0] else None,
        "nan_close": int(df["Close"].isna().sum()),
        "EMA20": float(df["EMA20"].iloc[-1]),
        "EMA50": float(df["EMA50"].iloc[-1]),
        "ATR14": float(df["ATR14"].iloc[-1]),
        "ADX14": float(df["ADX14"].iloc[-1]),
        "MOM63": float(mom),
        "DollarVol20": float(df["DollarVol20"].iloc[-1]) if not math.isnan(df["DollarVol20"].iloc[-1]) else 0.0,
        "long_ok": bool(long_ok and liquid and df.shape[0] >= 100),
        "short_ok": bool(short_ok and liquid and df.shape[0] >= 100),
        "liquid": bool(liquid),
        "reason_excluded": reason_excluded,
        "score_long": float(score_long),
        "score_short": float(score_short),
        "last_close": float(df["Close"].iloc[-1])
    }

def get_live_price(ticker: str, fallback: float) -> float:
    if yf is None:
        return fallback
    try:
        info = yf.Ticker(ticker).fast_info
        p = getattr(info, "last_price", None)
        if p is None or np.isnan(p):
            return fallback
        return float(p)
    except Exception:
        return fallback

def round_price(p: float) -> float:
    if p is None or math.isnan(p):
        return p
    if p >= 200:
        return round(p, 2)
    if p >= 20:
        return round(p, 2)
    if p >= 1:
        return round(p, 3)
    return round(p, 4)

def compute_levels(side: str, price: float, atr: float) -> Dict[str, float]:
    # TG se asume como precio de entrada actual (live)
    if side == "long":
        tg = price
        sl = price - K_SL_ATR * atr
        tp1 = price + K_TP1_ATR * atr
        tp2 = price + K_TP2_ATR * atr
        tl  = price - K_TL_ATR  * atr  # trailing start
    else:
        tg = price
        sl = price + K_SL_ATR * atr
        tp1 = price - K_TP1_ATR * atr
        tp2 = price - K_TP2_ATR * atr
        tl  = price + K_TL_ATR  * atr  # trailing start

    return {
        "TG": round_price(tg),
        "SL": round_price(sl),
        "TP1": round_price(tp1),
        "TP2": round_price(tp2),
        "TL": round_price(tl),
    }

# -------------- Ranking logic -------------
def build_candidates(metrics: List[Dict[str, Any]]) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
    strict: List[Dict[str,Any]] = []  # cumplen todo (para Top3)
    relaxed: List[Dict[str,Any]] = [] # tendencia+liquidez, ADX puede ser menor (para completar Top50)
    for m in metrics:
        # Log por ticker
        _log("DATA_AUDIT", {
            "ticker": m["ticker"], "rows": m["rows"], "start": m["start"], "end": m["end"],
            "nan_close": m["nan_close"], "reason_excluded": m["reason_excluded"]
        })
        # Strict inclusion
        if m["long_ok"] or m["short_ok"]:
            # elegir mejor lado si ambos (raro)
            side = None
            score = 0.0
            if m["long_ok"] and m["short_ok"]:
                if m["score_long"] >= m["score_short"]:
                    side, score = "long", m["score_long"]
                else:
                    side, score = "short", m["score_short"]
            elif m["long_ok"]:
                side, score = "long", m["score_long"]
            elif m["short_ok"]:
                side, score = "short", m["score_short"]

            strict.append({**m, "side": side, "score": float(score)})
        else:
            # relaxed: mantener tendencia y liquidez aunque ADX<M
            trend_ok = (m["EMA20"] > m["EMA50"] and m["MOM63"] > 0) or (m["EMA20"] < m["EMA50"] and m["MOM63"] < 0)
            if m["liquid"] and trend_ok and m["rows"] >= 100:
                side = "long" if (m["EMA20"] > m["EMA50"]) else "short"
                score = m["score_long"] if side=="long" else m["score_short"]
                relaxed.append({**m, "side": side, "score": float(score)})
    return strict, relaxed

def dedupe_by_ticker(cands: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    best: Dict[str, Dict[str,Any]] = {}
    for c in cands:
        t = c["ticker"]
        if t not in best or c["score"] > best[t]["score"]:
            best[t] = c
    return list(best.values())

def rank_and_split(strict: List[Dict[str,Any]], relaxed: List[Dict[str,Any]]) -> Tuple[List[str], List[Dict[str,Any]]]:
    # Dedupe por ticker y rank por score
    strict = sorted(dedupe_by_ticker(strict), key=lambda x: x["score"], reverse=True)
    relaxed = sorted(dedupe_by_ticker(relaxed), key=lambda x: x["score"], reverse=True)
    # Top50
    top50_cands = strict + [r for r in relaxed if r["ticker"] not in {c["ticker"] for c in strict}]
    top50_cands = top50_cands[:50]
    top50 = [c["ticker"] for c in top50_cands]
    # Top3 estrictos solamente
    top3 = strict[:3]
    return top50, top3

# -------------- Telegram ------------------
def _telegram_enabled() -> Tuple[bool, str]:
    if not ENABLE_TELEGRAM:
        return False, "disabled_via_env"
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False, "missing_env(TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID)"
    return True, "ok"

def _dedupe_token(top3: List[Dict[str,Any]]) -> str:
    basis = "|".join(f"{c['ticker']}:{c['side']}" for c in top3)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return hashlib.sha256(f"{basis}:{day}".encode()).hexdigest()

def _dedupe_should_send(token: str) -> bool:
    path = f"/tmp/top3_{token}.sent"
    return not os.path.exists(path)

def _dedupe_mark_sent(token: str):
    path = f"/tmp/top3_{token}.sent"
    try:
        with open(path, "w") as f:
            f.write("1")
    except Exception:
        pass

def _fmt_msg(top3: List[Dict[str,Any]]) -> str:
    lines = []
    hdr = "TOP 3 Señales (estrictas)"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for c in top3:
        t = c["ticker"]; side = c["side"].upper()
        px = c["live_price"]; atr = c["ATR14"]; adx = c["ADX14"]; mom = c["MOM63"]
        lv = c["levels"]
        rr1 = abs((lv["TP1"] - lv["TG"]) / (lv["TG"] - lv["SL"])) if lv["TG"] != lv["SL"] else float("nan")
        rr2 = abs((lv["TP2"] - lv["TG"]) / (lv["TG"] - lv["SL"])) if lv["TG"] != lv["SL"] else float("nan")
        dist_atr = abs((lv["TP1"] - px) / atr) if atr else float("nan")
        lines.append(
            f"Ticker: {t} | Lado: {side}\n"
            f"Precio: {round(px,2)} | ATR14: {round(atr,2)} | ADX14: {round(adx,1)} | Momentum63: {round(mom*100,1)}%\n"
            f"TG: {lv['TG']} | SL: {lv['SL']} | TP1: {lv['TP1']} | TP2: {lv['TP2']} | TL: {lv['TL']}\n"
            f"RR(TP1): {round(rr1,2)} | RR(TP2): {round(rr2,2)} | Dist a TP1: {round(dist_atr,2)} ATR\n"
            f"--"
        )
    return "\n".join(lines)

def notify_telegram(top3: List[Dict[str,Any]]) -> Dict[str, Any]:
    ok, why = _telegram_enabled()
    if not ok:
        return {"attempted": True, "enabled": False, "reason": why}
    import requests  # imported here to avoid dependency unless used
    msg = _fmt_msg(top3)
    token = _dedupe_token(top3)
    allowed = _dedupe_should_send(token)
    if not allowed:
        return {"attempted": True, "enabled": True, "reason": "deduped_skip", "dedupe_check": "skip"}
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}  # no parse_mode
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200 and resp.json().get("ok"):
            _dedupe_mark_sent(token)
            return {"attempted": True, "enabled": True, "status": 200, "response": "ok", "dedupe_check": "ok_to_send"}
        else:
            return {"attempted": True, "enabled": True, "reason": f"send_failed(status={resp.status_code} body={resp.text})", "dedupe_check": "ok_to_send"}
    except Exception as e:
        return {"attempted": True, "enabled": True, "reason": f"send_exception({type(e).__name__}: {e})", "dedupe_check": "ok_to_send"}

# -------------- Pipeline ------------------
def run_full_pipeline(audit: bool=False, **_ignored) -> Dict[str, Any]:
    t0 = time.time()
    as_of = datetime.now(timezone.utc).date().isoformat()
    try:
        tickers = load_universe()
        data = fetch_history(tickers)
        metrics: List[Dict[str,Any]] = []
        fetched = 0
        excluded = 0
        excluded_sample = []
        for t in tickers:
            df = data.get(t)
            if df is None or df.empty:
                _log("DATA_AUDIT", {"ticker": t, "rows": 0, "start": None, "end": None, "nan_close": None, "reason_excluded": "no_data"})
                excluded += 1
                if len(excluded_sample) < 5:
                    excluded_sample.append([t, "no_data"])
                continue
            enriched = enrich(df)
            m = latest_metrics(t, enriched)
            metrics.append(m)
            fetched += 1
            if m["reason_excluded"] and len(excluded_sample) < 5:
                excluded_sample.append([t, m["reason_excluded"]])

        _log("DATA_AUDIT", {"count": fetched})

        strict, relaxed = build_candidates(metrics)
        top50, top3_cands = rank_and_split(strict, relaxed)

        # Calcular niveles y motivos para top3 (reasons)
        final_top3 = []
        for c in top3_cands:
            live = get_live_price(c["ticker"], c["last_close"])
            lv = compute_levels(c["side"], live, c["ATR14"])
            reasons = []
            if c["side"] == "long":
                reasons.extend(["ema20>ema50" if c["EMA20"]>c["EMA50"] else "ema20<=ema50",
                                "momentum63_pos" if c["MOM63"]>0 else "momentum63<=0"])
            else:
                reasons.extend(["ema20<ema50" if c["EMA20"]<c["EMA50"] else "ema20>=ema50",
                                "momentum63_neg" if c["MOM63"]<0 else "momentum63>=0"])
            reasons.extend([f"adx>={MIN_ADX}", "avg_dollar_vol20_ok"])
            final_top3.append({
                "ticker": c["ticker"],
                "side": c["side"],
                "reasons": reasons,
                "live_price": float(live),
                "levels": lv,
                "ATR14": c["ATR14"],
                "ADX14": c["ADX14"],
                "MOM63": c["MOM63"],
            })

        # Notificación Telegram SOLO si hay 3 estrictos
        notified = False
        notify_info = {"attempted": False, "enabled": ENABLE_TELEGRAM}
        if len(final_top3) == 3:
            notify_info = notify_telegram(final_top3)
            notified = bool(notify_info.get("status") == 200 and notify_info.get("response") == "ok")

        # Logs resumen
        _log("RESUMEN", {"as_of": as_of, "top50": len(top50), "top3": len(final_top3)})
        took = round(time.time() - t0, 2)
        _log("app_evolutivo", {"msg": f"Pipeline listo · as_of={as_of} · top50={len(top50)} · top3_factors={len(final_top3)} · {took}s"})

        # Salida JSON
        return {
            "ok": True,
            "took_s": took,
            "as_of": as_of,
            "top50": top50,
            "top3_factors": [{"ticker": x["ticker"], "side": x["side"], "reasons": x["reasons"], "levels": x["levels"]} for x in final_top3],
            "diag": {
                "universe_count": len(tickers),
                "fetched_count": fetched,
                "excluded_count": len([m for m in metrics if m.get("reason_excluded")]),
                "excluded_sample": excluded_sample
            },
            "notified": notified,
            "notify_info": notify_info
        }
    except Exception as e:
        took = round(time.time() - t0, 2)
        _log("ERROR", {"type": type(e).__name__, "error": str(e)})
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "took_s": took, "as_of": as_of}

# Para pruebas locales rápidas
if __name__ == "__main__":
    out = run_full_pipeline(audit=True)
    print(json.dumps(out, indent=2))
