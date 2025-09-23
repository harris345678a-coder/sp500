import math
import datetime as dt
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np
import yfinance as yf

MIN_RR = float(__import__("os").getenv("PRESEND_MIN_RR", "1.8"))
ANCHOR_K_ATR = float(__import__("os").getenv("ANCHOR_K_ATR", "0.0"))  # usamos 0.0 porque anclamos al precio actual
STOP_K_ATR = float(__import__("os").getenv("STOP_K_ATR", "1.2"))
ATR_LEN = int(__import__("os").getenv("ATR_LEN", "14"))

def _atr(df: pd.DataFrame, n: int = ATR_LEN) -> float:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    atr = tr.rolling(n, min_periods=max(3, n//2)).mean().iloc[-1]
    return float(atr)

def _yf_hist(sym: str) -> pd.DataFrame:
    # 60m, fallback 1h
    for intr in ("60m", "1h"):
        df = yf.download(sym, period="90d", interval=intr, progress=False, auto_adjust=False, group_by=None, threads=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            # handle multiindex "Adj Close"
            if isinstance(df.columns, pd.MultiIndex):
                if ("Adj Close" in df.columns.get_level_values(-1)) and ("Close" not in df.columns.get_level_values(-1)):
                    # map Adj Close to Close for our calc
                    try:
                        sub = df.xs("Adj Close", axis=1, level=-1)
                        sub.columns = ["Adj Close"] if sub.shape[1] == 1 else [f"{c} Adj Close" for c in sub.columns]
                        # if single col, copy as Close
                        if "Adj Close" in sub.columns:
                            df = df.copy()
                            df["Close"] = sub["Adj Close"]
                    except Exception:
                        pass
                # flatten
                df.columns = [' '.join(map(str, c)).strip() if isinstance(c, tuple) else str(c) for c in df.columns]
            # Rename common variants
            ren = {}
            for col in list(df.columns):
                cl = str(col).strip().lower()
                if cl.endswith(" open"): ren[col] = "Open"
                elif cl.endswith(" high"): ren[col] = "High"
                elif cl.endswith(" low"): ren[col] = "Low"
                elif cl == "adj close": ren[col] = "Adj Close"
                elif cl == "close" or cl.endswith(" close"): ren[col] = "Close"
                elif cl == "volume" or cl.endswith(" volume"): ren[col] = "Volume"
            if ren:
                df = df.rename(columns=ren)
            for need in ("Open","High","Low","Close"):
                if need not in df.columns:
                    return pd.DataFrame()
            return df.dropna(subset=["Open","High","Low","Close"])
    return pd.DataFrame()

def _decide_side_and_strategy(c: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    adx = c.get("adx", 0.0)
    pdi = c.get("plus_di", 0.0)
    mdi = c.get("minus_di", 0.0)
    rsi15 = c.get("rsi15", 50.0)
    rsi60 = c.get("rsi60", 50.0)
    rsi1d = c.get("rsi1d", 50.0)

    # Side
    side = None
    if c.get("rsi_align_long") or (pdi > mdi and rsi60 >= 55 and rsi1d >= 55):
        side = "long"
    elif c.get("rsi_align_short") or (mdi > pdi and rsi60 <= 45 and rsi1d <= 45):
        side = "short"

    # Strategy
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
    # anclamos entry = price actual
    entry = price
    if side == "long":
        stop = entry - STOP_K_ATR * atr
        target = entry + max(min_rr * (entry - stop), 1.5 * atr)
        rr = (target - entry) / (entry - stop)
    else:
        stop = entry + STOP_K_ATR * atr
        target = entry - max(min_rr * (stop - entry), 1.5 * atr)
        rr = (entry - target) / (stop - entry)
    return {
        "trigger": round(entry, 4),
        "sl": round(stop, 4),
        "tp": round(target, 4),
        "rr": round(rr, 2),
        "atr": round(atr, 6),
    }

def build_top3_signals(top3_factors: List[Dict[str, Any]], as_of: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Devuelve approved_top3 (máx 3) y rejected_top3 con razones.
    Siempre intenta devolver 3 aprobadas re-anclando y redimensionando SL/TP.
    """
    cands = sorted(top3_factors, key=lambda x: x.get("rank_score", 0), reverse=True)[:5]  # miramos hasta 5 por si alguna falla datos
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

        price = float(df["Close"].dropna().iloc[-1])
        atr = _atr(df)

        sized = _size_signal(side, price, atr, MIN_RR)

        # Presend checks mínimos (ADX, RCI, MFI, RR)
        if c.get("adx", 0) < 20:
            rejected.append({"symbol": c.get("code", sym), "reason": "ADX < 20"})
            continue
        if c.get("rci", 0) < 0.85:
            rejected.append({"symbol": c.get("code", sym), "reason": "RCI bajo"})
            continue
        if c.get("mfi", 0) < 55:
            rejected.append({"symbol": c.get("code", sym), "reason": "MFI bajo"})
            continue
        if sized["rr"] < MIN_RR:
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

    # Si no llegamos a 3, agregamos los mejores rechazados como "needs_review" (nunca más de 3 total)
    i = 0
    while len(approved) < 3 and i < len(cands):
        c = cands[i]
        sym = c.get("ysymbol") or c.get("code")
        # Evita duplicados por código
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
