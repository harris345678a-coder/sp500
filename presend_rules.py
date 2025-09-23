
"""
presend_rules.py
--------------
Reglas de pre-envío (Pre-send) robustas para construir señales finales del Top 3.
- Sin parches, sin duplicados. Manejo estricto de errores.
- Anclaje con tolerancia (drift) de 10 bps cuando aplica.
- Fallbacks defensivos: si falla 60m, usa 1d para ATR; si falta algún indicador, no rompe.
- Evita errores de tipos/NaN/tz.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Dependencia interna del proyecto
try:
    from data_backend import DataBackend, DataError
except Exception:  # pragma: no cover
    # Fallback de nombres para evitar ImportError en tiempo de import
    class DataError(Exception):
        pass
    class DataBackend:
        def history(self, symbol: str, interval: str, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None):
            raise DataError("DataBackend no disponible en este entorno")
        def history_bulk(self, *args, **kwargs):
            raise DataError("DataBackend no disponible en este entorno")

# ---------------------------- utilidades seguras ----------------------------

def _is_num(x) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False

def _coerce_float(x, default: float = np.nan) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default

def _last_price(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty:
        return None
    series = df.get("Close")
    if series is None:
        return None
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])

def _high_n(df: pd.DataFrame, n: int) -> Optional[float]:
    if df is None or df.empty or "High" not in df.columns:
        return None
    s = pd.to_numeric(df["High"], errors="coerce").tail(n).dropna()
    if s.empty:
        return None
    return float(s.max())

def _low_n(df: pd.DataFrame, n: int) -> Optional[float]:
    if df is None or df.empty or "Low" not in df.columns:
        return None
    s = pd.to_numeric(df["Low"], errors="coerce").tail(n).dropna()
    if s.empty:
        return None
    return float(s.min())

def _true_range(o, h, l, c_prev):
    return max(h - l, abs(h - c_prev), abs(l - c_prev))

def _atr(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """
    ATR clásico sobre OHLC con cierre previo. Tolera gaps y NaN.
    """
    if df is None or df.empty:
        return None
    need_cols = {"Open", "High", "Low", "Close"}
    if any(col not in df.columns for col in need_cols):
        return None
    nd = df.copy()
    for col in need_cols:
        nd[col] = pd.to_numeric(nd[col], errors="coerce")
    nd = nd.dropna(subset=["Open", "High", "Low", "Close"])
    if len(nd) < period + 1:
        return None
    c_prev = nd["Close"].shift(1)
    tr = np.maximum.reduce([
        nd["High"] - nd["Low"],
        (nd["High"] - c_prev).abs(),
        (nd["Low"] - c_prev).abs(),
    ])
    atr = tr.rolling(window=period, min_periods=period).mean().iloc[-1]
    if not np.isfinite(atr):
        return None
    return float(atr)

# ---------------------------- modelo de señal ----------------------------

@dataclass
class Candidate:
    symbol: str
    ysymbol: str
    typ: str
    f: Dict

@dataclass
class BuiltSignal:
    code: str
    ysymbol: str
    type: str
    strategy: str
    side: str
    trigger: Optional[float]
    sl: Optional[float]
    tp: Optional[float]
    rr: Optional[float]
    atr: Optional[float]
    as_of: Optional[str]
    note: Optional[str] = None

    def to_dict(self) -> Dict:
        d = {
            "code": self.code,
            "ysymbol": self.ysymbol,
            "type": self.type,
            "strategy": self.strategy,
            "side": self.side,
            "trigger": self.trigger,
            "sl": self.sl,
            "tp": self.tp,
            "rr": self.rr,
            "atr": self.atr,
            "as_of": self.as_of,
        }
        if self.note:
            d["note"] = self.note
        return d

# ---------------------------- lógica de decisión ----------------------------

def _pick_side_and_strategy(f: Dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Devuelve (side, strategy, reason_if_none). Usa heurística robusta y los flags del ranking.
    """
    # Flags pre-computados por el pipeline
    rsi_long_ok = bool(f.get("rsi_align_long") or f.get("rsi_align_long_ok"))
    rsi_short_ok = bool(f.get("rsi_align_short") or f.get("rsi_align_short_ok"))
    adx_ok = bool(f.get("adx_ok"))
    adx = _coerce_float(f.get("adx"), np.nan)
    rsi4h = _coerce_float(f.get("rsi4h"), np.nan)
    rsi1d = _coerce_float(f.get("rsi1d"), np.nan)

    # Si ya hay dirección clara por flags
    if rsi_long_ok and adx_ok:
        return "long", "momentum", None
    if rsi_short_ok and adx_ok:
        return "short", "momentum", None

    # Fallback profesional (no bloquea por indeterminación leve)
    if np.isfinite(adx) and adx >= 25:
        if np.isfinite(rsi4h) and np.isfinite(rsi1d):
            if rsi4h >= 55 and rsi1d >= 55:
                return "long", "momentum", None
            if rsi4h <= 45 and rsi1d <= 45:
                return "short", "momentum", None

    # Si no se logra determinar, lo marcamos
    return None, None, "estrategia/side indeterminado"

def _risk_levels(side: str, price: float, atr: float) -> Tuple[float, float, float]:
    """
    Niveles con ATR 60m:
    - SL = 1.2 * ATR contra la dirección
    - TP = 1.9 * ATR a favor
    - Trigger = price + 0.30 * ATR en long (o -0.30 en short)
    """
    atr = float(atr)
    if side == "long":
        sl = price - 1.2 * atr
        tp = price + 1.9 * atr
        trigger = price + 0.30 * atr
    else:
        sl = price + 1.2 * atr
        tp = price - 1.9 * atr
        trigger = price - 0.30 * atr
    rr = abs((tp - price) / (price - sl)) if (price != sl) else np.nan
    return trigger, sl, tp, float(rr)

def _passes_orderbook(f: Dict) -> Tuple[bool, Optional[str]]:
    """
    Usa el flag precomputado 'ob_pass' si existe; si no, no bloquea.
    """
    ob = f.get("ob_pass")
    if ob is None:
        return True, None
    return bool(ob), None if ob else "orderbook/flow no favorable"

def _passes_rci_mfi(f: Dict, side: str) -> Tuple[bool, Optional[str]]:
    rci = _coerce_float(f.get("rci"), np.nan)
    mfi = _coerce_float(f.get("mfi"), np.nan)
    rci4h = _coerce_float(f.get("rci4h"), np.nan)
    mfi4h = _coerce_float(f.get("mfi4h"), np.nan)

    # Reglas suaves para no bloquear por ruido
    if side == "long":
        if np.isfinite(rci4h) and rci4h < 40:
            return False, "RCI 4H débil"
        if np.isfinite(mfi4h) and mfi4h < 40:
            return False, "MFI 4H débil"
    else:
        if np.isfinite(rci4h) and rci4h > 60:
            return False, "RCI 4H fuerte contra short"
        if np.isfinite(mfi4h) and mfi4h > 60:
            return False, "MFI 4H fuerte contra short"
    return True, None

def _anchor_ok(trigger: float, anchor: Optional[float], drift_bps: float = 10.0) -> Tuple[bool, Optional[str]]:
    """
    Acepta un drift de hasta X bps entre trigger y anchor.
    Si no hay anchor, no bloquea.
    """
    if anchor is None or not _is_num(anchor):
        return True, None
    anchor = float(anchor)
    if not _is_num(trigger):
        return False, "trigger inválido"
    tol = anchor * (drift_bps / 10000.0)  # bps a proporción
    if abs(trigger - anchor) <= tol:
        return True, None
    return False, f"breakout y anchoring incompatibles (|Δ|>{drift_bps}bps)"

# ---------------------------- core API ----------------------------

def build_top3_signals(top3_factors: List[Dict], as_of: Optional[str] = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Devuelve (approved, rejected).
    Cada aprobado incluye: strategy, side, trigger, sl, tp, rr, atr, as_of.
    """
    backend = DataBackend()
    approved: List[Dict] = []
    rejected: List[Dict] = []

    for f in top3_factors or []:
        code = f.get("code") or f.get("ysymbol") or f.get("symbol")
        ysymbol = f.get("ysymbol") or code
        typ = f.get("type") or "equity"
        cand = Candidate(symbol=code, ysymbol=ysymbol, typ=typ, f=f)

        # 1) Dirección y estrategia
        side, strategy, why = _pick_side_and_strategy(f)
        if side is None or strategy is None:
            rejected.append({"symbol": code, "reason": why or "estrategia/side indeterminado"})
            continue

        # 2) Datos recientes (60m primero, 1d fallback)
        df_60 = None
        atr_val = None
        price = None
        try:
            df_60 = backend.history(cand.symbol, "60m")
            price = _last_price(df_60)
            atr_val = _atr(df_60, period=14)
        except Exception as e:
            df_60 = None

        if (atr_val is None or not _is_num(atr_val)) or (price is None or not _is_num(price)):
            # Fallback 1d
            try:
                df_1d = backend.history(cand.symbol, "1d")
                if price is None:
                    price = _last_price(df_1d)
                if atr_val is None:
                    atr_val = _atr(df_1d, period=14)
            except Exception:
                pass

        if price is None or not _is_num(price):
            rejected.append({"symbol": code, "reason": "precio no disponible"})
            continue

        if atr_val is None or not _is_num(atr_val) or float(atr_val) <= 0:
            rejected.append({"symbol": code, "reason": "ATR no disponible"})
            continue

        price = float(price)
        atr_val = float(atr_val)

        # 3) Niveles RR
        trigger, sl, tp, rr = _risk_levels(side, price, atr_val)
        if not _is_num(rr) or rr < 1.2:
            rejected.append({"symbol": code, "reason": "RR insuficiente"})
            continue

        # 4) Orderbook / Flow (si está disponible en f)
        ok_ob, why_ob = _passes_orderbook(f)
        if not ok_ob:
            rejected.append({"symbol": code, "reason": why_ob})
            continue

        # 5) RCI/MFI 4H coherentes con el side
        ok_mom, why_mom = _passes_rci_mfi(f, side)
        if not ok_mom:
            rejected.append({"symbol": code, "reason": why_mom})
            continue

        # 6) Anclaje opcional (si f trae anchor)
        anchor = f.get("anchor")
        ok_anchor, why_anchor = _anchor_ok(trigger, anchor, drift_bps=10.0)
        if not ok_anchor:
            rejected.append({"symbol": code, "reason": why_anchor})
            continue

        built = BuiltSignal(
            code=cand.symbol,
            ysymbol=cand.ysymbol,
            type=cand.typ,
            strategy=strategy,
            side=side,
            trigger=round(trigger, 4),
            sl=round(sl, 4),
            tp=round(tp, 4),
            rr=round(rr, 2) if _is_num(rr) else None,
            atr=round(atr_val, 6),
            as_of=as_of,
        )
        approved.append(built.to_dict())

    return approved, rejected
