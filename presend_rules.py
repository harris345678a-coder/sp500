import math
import datetime as dt
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np
import yfinance as yf

MIN_RR = float(__import__("os").getenv("PRESEND_MIN_RR", "1.8"))
STOP_K_ATR = float(__import__("os").getenv("STOP_K_ATR", "1.2"))
ATR_LEN = int(__import__("os").getenv("ATR_LEN", "14"))

# ---------------------- Utilities ----------------------

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [' '.join(map(str, c)).strip() for c in df.columns]
    return df

def _canonical_field(col: str) -> Optional[str]:
    s = str(col).strip().lower()
    if s.endswith(' adj close') or s == 'adj close' or 'adjusted close' in s:
        return 'Adj Close'
    if s.endswith(' open') or s == 'open':
        return 'Open'
    if s.endswith(' high') or s == 'high':
        return 'High'
    if s.endswith(' low') or s == 'low':
        return 'Low'
    if s.endswith(' close') or s == 'close':
        return 'Close'
    if s.endswith(' volume') or s == 'volume':
        return 'Volume'
    return None

def _select_best(series_like) -> Optional[pd.Series]:
    """If given a DataFrame with multiple candidate columns, pick the one with most non-nulls."""
    if isinstance(series_like, pd.Series):
        return series_like
    if isinstance(series_like, pd.DataFrame):
        if series_like.shape[1] == 1:
            return series_like.iloc[:, 0]
        counts = series_like.notna().sum()
        # counts is Series indexed by column; pick max
        best_col = counts.idxmax()
        return series_like[best_col]
    return None

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = _flatten(df)

    # Map all possible variants to canonical keys, then select best per key
    buckets: Dict[str, list] = {'Open':[], 'High':[], 'Low':[], 'Close':[], 'Adj Close':[], 'Volume':[]}
    for col in df.columns:
        canon = _canonical_field(col)
        if canon in buckets:
            s = df[col]
            if isinstance(s, pd.DataFrame):
                for c in s.columns:
                    buckets[canon].append(pd.to_numeric(s[c], errors='coerce'))
            else:
                buckets[canon].append(pd.to_numeric(s, errors='coerce'))

    out: Dict[str, pd.Series] = {}
    for key in ('Open','High','Low','Close','Adj Close','Volume'):
        if buckets[key]:
            # pick the one with most data
            counts = [ser.notna().sum() for ser in buckets[key]]
            best = int(np.argmax(counts))
            out[key] = buckets[key][best]

    # Fallback Close -> Adj Close
    if 'Close' not in out or out['Close'].isna().all():
        if 'Adj Close' in out and not out['Adj Close'].isna().all():
            out['Close'] = out['Adj Close']
        else:
            return pd.DataFrame()

    res = pd.DataFrame({k: v for k, v in out.items() if k in ('Open','High','Low','Close','Volume')})
    if res.empty:
        return res
    # Drop rows missing core prices
    res = res.dropna(subset=['Open','High','Low','Close'])
    # Ensure DatetimeIndex
    if not isinstance(res.index, pd.DatetimeIndex):
        try:
            res.index = pd.to_datetime(res.index, utc=False)
        except Exception:
            pass
    return res.sort_index()

def _yf_hist(sym: str) -> pd.DataFrame:
    for intr in ("60m", "1h"):
        df = yf.download(sym, period="90d", interval=intr, progress=False, auto_adjust=False, group_by=None, threads=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            nd = _normalize_ohlcv(df)
            if not nd.empty and all(c in nd.columns for c in ("Open","High","Low","Close")):
                return nd
    return pd.DataFrame()

def _atr(df: pd.DataFrame, n: int = ATR_LEN) -> float:
    # Ensure single Series per field
    h = _select_best(df.get("High"))
    l = _select_best(df.get("Low"))
    c = _select_best(df.get("Close"))
    if h is None or l is None or c is None:
        return float("nan")
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    atr = tr.rolling(n, min_periods=max(3, n//2)).mean().dropna()
    return float(atr.iloc[-1]) if not atr.empty else float("nan")

# ---------------------- Strategy/Sizing ----------------------

def _decide_side_and_strategy(c: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    adx = c.get("adx", 0.0)
    pdi = c.get("plus_di", 0.0)
    mdi = c.get("minus_di", 0.0)
    rsi15 = c.get("rsi15", 50.0)
    rsi60 = c.get("rsi60", 50.0)
    rsi1d = c.get("rsi1d", 50.0)

    side = None
    if c.get("rsi_align_long") or (pdi > mdi and rsi60 >= 55 and rsi1d >= 55):
        side = "long"
    elif c.get("rsi_align_short") or (mdi > pdi and rsi60 <= 45 and rsi1d <= 45):
        side = "short"

    strat = None
    if adx >= 25:
        if side == "long" and rsi60 >= 60 and rsi1d >= 60:
            strat = "breakout-continuation"
        elif side == "short" and rsi60 <= 40 and rsi1d <= 40:
            strat = "breakdown-continuation"
    if strat is None:
        if side == "long" and rsi15 <= 35 and 45 <= rsi60 <= 60:
            strat = "reversion-long"
        elif side == "short" and rsi15 >= 65 and 40 <= rsi60 <= 55:
            strat = "reversion-short"
    if strat is None and adx >= 20 and c.get("rci", 0.0) >= 0.9:
        strat = "momentum"
    return side, strat

def _size_signal(side: str, price: float, atr: float, min_rr: float = MIN_RR) -> Dict[str, float]:
    entry = price  # anchor at last price
    if side == "long":
        stop = entry - STOP_K_ATR * atr
        target = entry + max(min_rr * (entry - stop), 1.5 * atr)
        rr = (target - entry) / (entry - stop) if (entry - stop) > 0 else float("nan")
    else:
        stop = entry + STOP_K_ATR * atr
        target = entry - max(min_rr * (stop - entry), 1.5 * atr)
        rr = (entry - target) / (stop - entry) if (stop - entry) > 0 else float("nan")
    return {
        "trigger": round(float(entry), 4),
        "sl": round(float(stop), 4),
        "tp": round(float(target), 4),
        "rr": round(float(rr), 2) if not math.isnan(rr) else None,
        "atr": round(float(atr), 6) if not math.isnan(atr) else None,
    }

# ---------------------- Main ----------------------

def build_top3_signals(top3_factors: List[Dict[str, Any]], as_of: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cands = sorted(top3_factors, key=lambda x: x.get("rank_score", 0), reverse=True)[:5]
    approved: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for c in cands:
        sym = c.get("ysymbol") or c.get("code")
        if not sym:
            rejected.append({"symbol": str(c.get("code")), "reason": "sin símbolo"})
            continue

        side, strat = _decide_side_and_strategy(c)
        if side is None or strat is None:
            rejected.append({"symbol": c.get("code", sym), "reason": "estrategia/side indeterminado"})
            continue

        df = _yf_hist(sym)
        if df.empty or "Close" not in df.columns:
            rejected.append({"symbol": c.get("code", sym), "reason": "datos intradía insuficientes"})
            continue

        close_obj = df["Close"]
        close_ser = _select_best(close_obj)
        if close_ser is None or close_ser.dropna().empty:
            rejected.append({"symbol": c.get("code", sym), "reason": "Close inválido"})
            continue

        price = float(close_ser.dropna().iloc[-1])
        atr = _atr(df)
        if math.isnan(atr) or atr <= 0:
            rejected.append({"symbol": c.get("code", sym), "reason": "ATR inválido"})
            continue

        sized = _size_signal(side, price, atr, MIN_RR)

        # Presend checks mínimos
        if c.get("adx", 0) < 20:
            rejected.append({"symbol": c.get("code", sym), "reason": "ADX < 20"})
            continue
        if c.get("rci", 0) < 0.85:
            rejected.append({"symbol": c.get("code", sym), "reason": "RCI bajo"})
            continue
        if c.get("mfi", 0) < 55:
            rejected.append({"symbol": c.get("code", sym), "reason": "MFI bajo"})
            continue
        if sized["rr"] is None or sized["rr"] < MIN_RR:
            rejected.append({"symbol": c.get("code", sym), "reason": f"RR {sized['rr']} < {MIN_RR}"})
            continue

        approved.append({
            "code": c.get("code"),
            "ysymbol": sym,
            "type": c.get("type"),
            "strategy": strat,
            "side": side,
            **sized,
            "as_of": as_of,
        })
        if len(approved) == 3:
            break

    # Completar hasta 3 con 'needs_review' si fuese necesario (sin duplicados)
    i = 0
    while len(approved) < 3 and i < len(cands):
        c = cands[i]
        sym = c.get("ysymbol") or c.get("code")
        if any(a.get("code") == c.get("code") for a in approved):
            i += 1
            continue
        approved.append({
            "code": c.get("code"),
            "ysymbol": sym,
            "type": c.get("type"),
            "strategy": "needs_review",
            "side": "long" if c.get("rsi_align_long") else ("short" if c.get("rsi_align_short") else "long"),
            "trigger": None, "sl": None, "tp": None, "rr": None,
            "as_of": as_of,
            "note": "No pasó presend duro; marcada como needs_review"
        })
        i += 1

    return approved[:3], rejected
