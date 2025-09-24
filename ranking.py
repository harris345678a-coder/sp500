# -*- coding: utf-8 -*-
"""
Archivo: ranking.py
(versión robusta con universo dinámico, descarga en lote y filtros estrictos)
"""
from __future__ import annotations
import os, time, math, json, logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

__all__ = ["run_full_pipeline", "get_universe"]

# ---- Logging ----
def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None: return default
    return str(val).strip().lower() in ("1","true","yes","y","on")

def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("ranking")
    level_name = os.getenv("RANKING_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | ranking | %(message)s"))
        logger.addHandler(h)
        logger.propagate = False
    yf_debug = _bool_env("YFINANCE_DEBUG", False)
    yf_logger = logging.getLogger("yfinance")
    yf_logger.setLevel(logging.DEBUG if yf_debug else logging.INFO)
    if yf_debug and not yf_logger.handlers:
        yh = logging.StreamHandler()
        yh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | yfinance | %(message)s"))
        yf_logger.addHandler(yh); yf_logger.propagate = False
    return logger

log = _setup_logging()
DATA_AUDIT_ON = _bool_env("RANKING_DATA_AUDIT", True)
def _log_audit(context: str, payload: Dict):
    if DATA_AUDIT_ON and log.isEnabledFor(logging.INFO):
        if context:
            log.info("DATA_AUDIT | %s | %s", context, json.dumps(payload, ensure_ascii=False))
        else:
            log.info("DATA_AUDIT | %s", json.dumps(payload, ensure_ascii=False))

# ---- Universo ----
UNIVERSE_BASE = [
    "SPY","QQQ","IWM","XLK","SOXX","SMH","GDX","GLD","SLV","DBC","DBA","SHY",
    "AAPL","NVDA","AMZN","GC=F","CL=F"
]

def _safe_name(s: str) -> str: return s.strip()

def _load_universe_from_env_inline() -> Optional[List[str]]:
    raw = os.getenv("RANKING_UNIVERSE")
    if not raw: return None
    out, seen = [], set()
    for part in raw.split(","):
        t = _safe_name(part)
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out or None

def _load_universe_from_csv(path: str) -> Optional[List[str]]:
    try:
        df = pd.read_csv(path)
        col = None
        for c in df.columns:
            if str(c).lower() in ("symbol","ticker","symbols","tickers"): col=c; break
        if col is None: col = df.columns[0]
        vals = [ _safe_name(str(x)) for x in df[col].dropna().astype(str) ]
        out, seen = [], set()
        for v in vals:
            if v and v not in seen:
                seen.add(v); out.append(v)
        return out or None
    except Exception as e:
        log.warning("UNIVERSE_CSV_FAIL | %s | %r", path, e); return None

def _load_universe_from_url(url: str) -> Optional[List[str]]:
    try:
        df = pd.read_csv(url)
        col = None
        for c in df.columns:
            if str(c).lower() in ("symbol","ticker","symbols","tickers"): col=c; break
        if col is None: col = df.columns[0]
        vals = [ _safe_name(str(x)) for x in df[col].dropna().astype(str) ]
        out, seen = [], set()
        for v in vals:
            if v and v not in seen:
                seen.add(v); out.append(v)
        return out or None
    except Exception as e:
        log.warning("UNIVERSE_URL_FAIL | %s | %r", url, e); return None

def get_universe() -> List[str]:
    env_list = _load_universe_from_env_inline()
    if env_list:
        _log_audit("universe_source", {"source":"RANKING_UNIVERSE","count":len(env_list)})
        return env_list
    path = os.getenv("RANKING_UNIVERSE_CSV_PATH")
    if path:
        lst = _load_universe_from_csv(path)
        if lst:
            _log_audit("universe_source", {"source":"RANKING_UNIVERSE_CSV_PATH","path":path,"count":len(lst)})
            return lst
    url = os.getenv("RANKING_UNIVERSE_CSV_URL")
    if url:
        lst = _load_universe_from_url(url)
        if lst:
            _log_audit("universe_source", {"source":"RANKING_UNIVERSE_CSV_URL","url":url,"count":len(lst)})
            return lst
    # fallback
    out, seen = [], set()
    for s in UNIVERSE_BASE:
        s = _safe_name(s)
        if s and s not in seen:
            seen.add(s); out.append(s)
    _log_audit("universe_source", {"source":"UNIVERSE_BASE","count":len(out)})
    return out

# ---- Parámetros / Filtros ----
@dataclass
class FetchParams:
    period: str = "2y"
    interval: str = "1d"
    auto_adjust: bool = True
    actions: bool = True
    min_rows: int = 200
    max_stale_days: int = 7

FETCH = FetchParams()

@dataclass
class Filters:
    require_trend_up: bool = True
    min_price: float = 1.0
    min_dollar_vol_20d: float = 5e6

FILTERS = Filters()
TOP50_CAP = 50

# ---- Utils ----
def _normalize_history_df(df: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.copy(); df["Close"] = df["Adj Close"]
    return df

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _highest(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).max()

def _dollar_volume(df: pd.DataFrame) -> pd.Series:
    if "Close" in df.columns and "Volume" in df.columns:
        return df["Close"] * df["Volume"].fillna(0)
    return pd.Series([0]*len(df), index=df.index)

# ---- Fetch ----
def _fetch_bulk(symbols: List[str], params: FetchParams) -> Dict[str, pd.DataFrame]:
    if not symbols: return {}
    try:
        data = yf.download(
            tickers=" ".join(symbols),
            period=params.period, interval=params.interval,
            auto_adjust=params.auto_adjust, actions=params.actions,
            group_by="ticker", threads=True, progress=False,
        )
        out = {}
        if isinstance(data.columns, pd.MultiIndex):
            for sym in symbols:
                if sym in data.columns.get_level_values(0):
                    sub = data[sym].dropna(how="all")
                    if not sub.empty:
                        out[sym] = _normalize_history_df(sub)
        else:
            sub = data.dropna(how="all")
            if not sub.empty:
                out[symbols[0]] = _normalize_history_df(sub)
        return out
    except Exception as e:
        log.warning("BULK_DOWNLOAD_FAIL | %d symbols | %r", len(symbols), e); return {}

def _fetch_single(symbol: str, params: FetchParams) -> Optional[pd.DataFrame]:
    tries, backoff = 3, 0.8
    for i in range(1, tries+1):
        try:
            df = yf.Ticker(symbol).history(
                period=params.period, interval=params.interval,
                auto_adjust=params.auto_adjust, actions=params.actions,
            )
            if df is None or df.empty: raise RuntimeError("history vacío")
            df = _normalize_history_df(df)
            for col in ("Open","High","Low","Close"): 
                if col not in df.columns: raise RuntimeError(f"columna faltante: {col}")
            return df
        except Exception as e:
            if i == tries:
                log.warning("FETCH_FAIL | %s | intento=%s/%s | %r", symbol, i, tries, e); return None
            time.sleep(backoff*i)

# ---- Filtros / Señales ----
def _apply_mandatory_filters(symbol: str, df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    if len(df) < FETCH.min_rows: return False, f"min_rows<{FETCH.min_rows}"
    last_dt = df.index[-1]
    if hasattr(last_dt, "to_pydatetime"): last_dt = last_dt.to_pydatetime()
    if datetime.now(timezone.utc) - last_dt.replace(tzinfo=timezone.utc) > timedelta(days=FETCH.max_stale_days):
        return False, "stale_data"
    last_close = float(df["Close"].iloc[-1])
    if last_close < FILTERS.min_price: return False, f"price<{FILTERS.min_price}"
    if FILTERS.require_trend_up:
        ema20 = _ema(df["Close"], 20); ema50 = _ema(df["Close"], 50)
        if not bool(ema20.iloc[-1] > ema50.iloc[-1]): return False, "ema20<=ema50"
    dv = _dollar_volume(df).rolling(20).mean()
    if dv.iloc[-1] < FILTERS.min_dollar_vol_20d: return False, f"dollar_vol20<{int(FILTERS.min_dollar_vol_20d)}"
    return True, None

def _compute_signals(symbol: str, df: pd.DataFrame) -> List[str]:
    reasons = []
    hi55 = _highest(df["Close"], 55)
    if not math.isnan(hi55.iloc[-1]) and df["Close"].iloc[-1] > hi55.iloc[-1]: reasons.append("breakout55")
    ema20 = _ema(df["Close"], 20); ema50 = _ema(df["Close"], 50)
    if ema20.iloc[-1] > ema50.iloc[-1]: reasons.append("ema20>ema50")
    if len(df) >= 63 and df["Close"].pct_change(63).iloc[-1] > 0: reasons.append("momentum63_pos")
    return reasons

def _score_for_top50(df: pd.DataFrame) -> float:
    c = df["Close"]
    r63 = c.pct_change(63).iloc[-1] if len(c) >= 63 else 0.0
    r126 = c.pct_change(126).iloc[-1] if len(c) >= 126 else r63
    return float(0.5*r63 + 0.5*r126)

# ---- Pipeline ----
def run_full_pipeline(*, audit: Optional[bool]=None, **_) -> Dict:
    t0 = time.time(); as_of = datetime.now().date().isoformat()
    universe = get_universe()
    _log_audit("universe", {"count": len(universe), "tickers_sample": universe[:10]})

    fetched, excluded = {}, {}
    top50_candidates = []
    top3 = []

    BULK_THRESHOLD = 30
    remaining = list(universe)

    if len(universe) >= BULK_THRESHOLD:
        bulk = _fetch_bulk(universe, FETCH)
        for sym, df in bulk.items():
            if sym in remaining: remaining.remove(sym)
            df = df.dropna(subset=["Close"]).copy()
            payload = {"ticker": sym, "rows": int(len(df)),
                       "start": str(df.index[0].date()) if len(df) else None,
                       "end": str(df.index[-1].date()) if len(df) else None,
                       "nan_close": int(df["Close"].isna().sum()), "reason_excluded": None}
            ok, reason = _apply_mandatory_filters(sym, df)
            if not ok: payload["reason_excluded"]=reason; excluded[sym]=reason or "filter_fail"
            else: fetched[sym]=df
            _log_audit("", payload)

    for sym in remaining:
        df = _fetch_single(sym, FETCH)
        if df is None or df.empty: excluded[sym]="fetch_fail_or_empty"; continue
        df = df.dropna(subset=["Close"]).copy()
        payload = {"ticker": sym, "rows": int(len(df)),
                   "start": str(df.index[0].date()) if len(df) else None,
                   "end": str(df.index[-1].date()) if len(df) else None,
                   "nan_close": int(df["Close"].isna().sum()), "reason_excluded": None}
        ok, reason = _apply_mandatory_filters(sym, df)
        if not ok: payload["reason_excluded"]=reason; excluded[sym]=reason or "filter_fail"
        else: fetched[sym]=df
        _log_audit("", payload)

    for sym, df in fetched.items():
        top50_candidates.append((sym, _score_for_top50(df)))
    top50_candidates.sort(key=lambda x: x[1], reverse=True)
    top50_list = [s for s,_ in top50_candidates[:50]]

    for sym, _ in top50_candidates:
        if len(top3) >= 3: break
        reasons = _compute_signals(sym, fetched[sym])
        if reasons: top3.append({"ticker": sym, "reasons": reasons})

    _log_audit("", {"count": len(top50_list)})
    _log_audit("", {"count": len(top3), "tickers": [s["ticker"] for s in top3]})
    log.info("RESUMEN | as_of=%s | top50=%d | top3=%d", as_of, len(top50_list), len(top3))

    diag = {"universe_count": len(universe),
            "fetched_count": len(fetched),
            "excluded_count": len(excluded),
            "excluded_sample": list({k:v for k,v in list(excluded.items())[:5]}.items())}

    return {"ok": True, "took_s": round(time.time()-t0,2), "as_of": as_of,
            "top50": top50_list, "top3_factors": top3, "diag": diag}

if __name__ == "__main__":
    out = run_full_pipeline()
    print(json.dumps(out, indent=2, ensure_ascii=False))
