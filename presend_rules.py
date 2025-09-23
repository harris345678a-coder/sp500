from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from data_backend import DataBackend, DataError

# ---------- helpers (robust OHLCV usage) ----------

def _last_price(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty:
        return None
    c = df.get("Close")
    if c is None:
        return None
    c = pd.to_numeric(c, errors="coerce").dropna()
    if c.empty:
        return None
    return float(c.iloc[-1])

def _atr(df: pd.DataFrame, n: int = 14) -> Optional[float]:
    if df is None or df.empty or "High" not in df or "Low" not in df or "Close" not in df:
        return None
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    atr = pd.Series(tr, index=df.index).rolling(n, min_periods=n).mean().dropna()
    if atr.empty:
        return None
    return float(atr.iloc[-1])

# ---------- gating rules (pre-send) ----------

def _rsi_alignment_ok(f: Dict) -> bool:
    # Require long xor short alignment; prefer long
    return bool(f.get("rsi_align_long") or f.get("rsi_align_short"))

def _trend_strength_ok(f: Dict) -> bool:
    # Accept if either 60m ADX or 4H ADX >= 25
    adx60 = float(f.get("adx") or 0.0)
    adx4 = float(f.get("adx4h") or 0.0)
    return (adx60 >= 25.0) or (adx4 >= 25.0)

def _momentum_quality_ok(f: Dict) -> bool:
    # Use RCI/MFI on 4H
    rci4 = float(f.get("rci4h") or -999)
    mfi4 = float(f.get("mfi4h") or -999)
    return (rci4 >= 0.0) and (mfi4 >= 55.0)

def _choose_strategy(f: Dict) -> str:
    # Simple, deterministic strategy picker
    rsi4 = float(f.get("rsi4h") or np.nan)
    adx4 = float(f.get("adx4h") or np.nan)
    rsi60 = float(f.get("rsi60") or np.nan)
    if np.isnan(rsi4) or np.isnan(adx4) or np.isnan(rsi60):
        return "needs_review"
    # Strong momentum long
    if (f.get("rsi_align_long") and rsi4 >= 60 and adx4 >= 25):
        return "momentum"
    # Breakout continuation
    if (f.get("rsi_align_long") and rsi60 >= 55 and rsi4 >= 55 and adx4 >= 20):
        return "breakout-continuation"
    # Potential short (not enabled here)
    if f.get("rsi_align_short"):
        return "needs_review"
    return "needs_review"

# ---------- order/levels builders ----------

def _levels_long_from_60m(df60: pd.DataFrame, last: float, atr: float) -> Tuple[float,float,float,float]:
    # trigger slightly above last close; SL = last - 1.2*ATR; TP = last + 2.16*ATR  (RR ~1.8)
    if atr is None or atr <= 0:
        # fallback small offsets
        atr = max(last * 0.003, 0.1)
    trigger = float(np.round(last + 0.25 * atr, 4))
    sl = float(np.round(last - 1.2 * atr, 4))
    tp = float(np.round(last + 2.16 * atr, 4))
    rr = float(np.round((tp - trigger) / max(trigger - sl, 1e-6), 2))
    return trigger, sl, tp, rr

def _rr_ok(rr: float) -> bool:
    return rr >= 1.6

def _anchoring_ok(last: float, trigger: float, atr: float) -> bool:
    # Distance last->trigger must be <= 0.6*ATR
    return abs(trigger - last) <= 0.6 * atr

# ---------- main: build_top3_signals ----------

def build_top3_signals(top3_factors: List[Dict], as_of: Optional[str] = None) -> Tuple[List[Dict], List[Dict]]:
    backend = DataBackend()
    approved: List[Dict] = []
    rejected: List[Dict] = []

    for f in top3_factors:
        sym = f["ysymbol"]
        # Data for price/ATR from 60m (stable granularity) and 15m for fine anchor
        d60 = backend.history(sym, "60m")
        if d60 is None or d60.empty:
            rejected.append({"symbol": sym, "reason": "sin datos 60m"})
            continue

        last = _last_price(d60)
        atr = _atr(d60, n=14)
        if last is None or atr is None:
            rejected.append({"symbol": sym, "reason": "precio/ATR no disponibles"})
            continue

        if not _rsi_alignment_ok(f):
            rejected.append({"symbol": sym, "reason": "RSI no alineado"})
            continue
        if not _trend_strength_ok(f):
            rejected.append({"symbol": sym, "reason": "ADX insuficiente"})
            continue
        if not _momentum_quality_ok(f):
            rejected.append({"symbol": sym, "reason": "RCI/MFI 4H débiles"})
            continue

        strategy = _choose_strategy(f)
        if strategy not in ("momentum", "breakout-continuation"):
            rejected.append({"symbol": sym, "reason": "estrategia/side indeterminado"})
            continue

        # Build levels
        trigger, sl, tp, rr = _levels_long_from_60m(d60, last, atr)

        if not _rr_ok(rr):
            rejected.append({"symbol": sym, "reason": "RR insuficiente"})
            continue
        if not _anchoring_ok(last, trigger, atr):
            rejected.append({"symbol": sym, "reason": "anchoring fuera de tolerancia"})
            continue

        approved.append({
            "code": f.get("code") or sym.replace("-", ""),
            "ysymbol": sym,
            "type": f.get("type", "equity"),
            "strategy": strategy,
            "side": "long",
            "trigger": float(np.round(trigger, 4)),
            "sl": float(np.round(sl, 4)),
            "tp": float(np.round(tp, 4)),
            "rr": float(np.round(rr, 2)),
            "atr": float(np.round(atr, 6)),
            "as_of": as_of or "",
        })

    # Guarantee exclusivity: none of the approved should be in rejected
    rej_filtered = [r for r in rejected if r.get("symbol") not in {a["ysymbol"] for a in approved}]
    return approved, rej_filtered
