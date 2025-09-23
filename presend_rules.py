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

def _swing_levels(df: pd.DataFrame, lookback: int = 20) -> Tuple[Optional[float], Optional[float]]:
    if df is None or df.empty:
        return None, None
    hi = pd.to_numeric(df["High"], errors="coerce").tail(lookback).dropna()
    lo = pd.to_numeric(df["Low"], errors="coerce").tail(lookback).dropna()
    if hi.empty or lo.empty:
        return None, None
    return float(hi.max()), float(lo.min())

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
    # Deterministic strategy picker
    rsi4 = float(f.get("rsi4h") or np.nan)
    adx4 = float(f.get("adx4h") or np.nan)
    rsi60 = float(f.get("rsi60") or np.nan)
    if np.isnan(rsi4) or np.isnan(adx4) or np.isnan(rsi60):
        return "needs_review"
    if (f.get("rsi_align_long") and rsi4 >= 60 and adx4 >= 25):
        return "momentum"
    if (f.get("rsi_align_long") and rsi60 >= 55 and rsi4 >= 55 and adx4 >= 20):
        return "breakout-continuation"
    if f.get("rsi_align_short"):
        return "needs_review"
    return "needs_review"

# ---------- Dynamic RR/anchoring with swing bounds ----------

def _solve_long_levels_dynamic(d60: pd.DataFrame, last: float, atr: float,
                               rr_target: float = 1.6,
                               anchor_limit_mult: float = 0.6,
                               swing_lb: int = 20) -> Tuple[float,float,float,float]:
    """
    Usa ATR en vivo y swings recientes para fijar niveles:
    - trigger >= swing_high + 0.1*ATR (breakout real)
    - SL <= swing_low - 0.2*ATR (bajo soporte)
    - |trigger - last| <= 0.6*ATR (anchoring)
    - RR >= rr_target ajustando TP de forma exacta
    """
    if atr is None or atr <= 0:
        atr = max(last * 0.003, 0.1)

    swing_high, swing_low = _swing_levels(d60, lookback=swing_lb)
    if swing_high is None or swing_low is None:
        # fallback: usa últimos valores directos
        swing_high = float(pd.to_numeric(d60["High"], errors="coerce").tail(5).max())
        swing_low = float(pd.to_numeric(d60["Low"], errors="coerce").tail(5).min())

    # SL por debajo de soporte
    sl = min(last - 1.0 * atr, swing_low - 0.2 * atr)

    # Trigger mínimo por breakout y por anclaje
    min_trigger_breakout = swing_high + 0.1 * atr
    max_trigger_anchor = last + anchor_limit_mult * atr
    if min_trigger_breakout > max_trigger_anchor:
        # No se puede cumplir breakout + anchoring al mismo tiempo
        # Dejar constancia via excepción para que la señal sea rechazada con razón clara
        raise DataError("breakout y anchoring incompatibles (mover trigger viola anclaje)")
    # Trigger inicial: lo más exigente entre breakout y delta RR
    # Fijamos TP exacto para RR: TP = trigger + rr_target*(trigger - SL)
    # Elegimos trigger = min( max_trigger_anchor, max(min_trigger_breakout, last) )
    trigger = float(np.round(min(max_trigger_anchor, max(min_trigger_breakout, last)), 4))

    # TP exacto por RR objetivo
    den = max(trigger - sl, 1e-9)
    tp = float(np.round(trigger + rr_target * den, 4))

    # RR real
    rr = float(np.round((tp - trigger) / max(trigger - sl, 1e-9), 2))

    return trigger, float(np.round(sl,4)), tp, rr

def _rr_ok(rr: float, rr_target: float = 1.6) -> bool:
    return rr >= rr_target

def _anchoring_ok(last: float, trigger: float, atr: float, anchor_limit_mult: float = 0.6) -> bool:
    return abs(trigger - last) <= anchor_limit_mult * atr

# ---------- main: build_top3_signals ----------

def build_top3_signals(top3_factors: List[Dict], as_of: Optional[str] = None) -> Tuple[List[Dict], List[Dict]]:
    backend = DataBackend()
    approved: List[Dict] = []
    rejected: List[Dict] = []

    for f in top3_factors:
        sym = f["ysymbol"]
        # Datos 60m en vivo
        try:
            d60 = backend.history(sym, "60m")
        except Exception as e:
            rejected.append({"symbol": sym, "reason": f"sin datos 60m: {e}"})
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

        try:
            trigger, sl, tp, rr = _solve_long_levels_dynamic(d60, last, atr, rr_target=1.6, anchor_limit_mult=0.6, swing_lb=20)
        except DataError as ex:
            rejected.append({"symbol": sym, "reason": str(ex)})
            continue

        if not _rr_ok(rr, rr_target=1.6):
            rejected.append({"symbol": sym, "reason": "RR insuficiente"})
            continue
        if not _anchoring_ok(last, trigger, atr, anchor_limit_mult=0.6):
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

    rej_filtered = [r for r in rejected if r.get("symbol") not in {a["ysymbol"] for a in approved}]
    return approved, rej_filtered
