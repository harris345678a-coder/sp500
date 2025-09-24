
# ranking.py - Pipeline de ranking y factores multi-TF (profesional, sin duplicados)

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_backend import DataBackend, DataError
from presend_rules import build_top3_signals

# ============================
# Universo base (símbolo->ysymbol y tipo)
# ============================

# Notas:
# - code: identificador interno estable (sin guiones)
# - ysymbol: como se consulta al vendor (ej: "BRK-B")
# - type: "equity" o "etf" (sólo informativo)
UNIVERSE: List[Dict[str, str]] = [
    {"code": "TSLA", "ysymbol": "TSLA", "type": "equity"},
    {"code": "NVDA", "ysymbol": "NVDA", "type": "equity"},
    {"code": "QQQ",  "ysymbol": "QQQ",  "type": "equity"},
    {"code": "SPY",  "ysymbol": "SPY",  "type": "equity"},
    {"code": "AAPL", "ysymbol": "AAPL", "type": "equity"},
    {"code": "GOOGL","ysymbol": "GOOGL","type": "equity"},
    {"code": "MSFT", "ysymbol": "MSFT", "type": "equity"},
    {"code": "AMZN", "ysymbol": "AMZN", "type": "equity"},
    {"code": "META", "ysymbol": "META", "type": "equity"},
    {"code": "IWM",  "ysymbol": "IWM",  "type": "equity"},
    {"code": "VOO",  "ysymbol": "VOO",  "type": "equity"},
    {"code": "IVV",  "ysymbol": "IVV",  "type": "equity"},
    {"code": "GLD",  "ysymbol": "GLD",  "type": "equity"},
    {"code": "TLT",  "ysymbol": "TLT",  "type": "equity"},
    {"code": "JPM",  "ysymbol": "JPM",  "type": "equity"},
    {"code": "SMH",  "ysymbol": "SMH",  "type": "equity"},
    {"code": "LQD",  "ysymbol": "LQD",  "type": "equity"},
    {"code": "XLK",  "ysymbol": "XLK",  "type": "equity"},
    {"code": "GDX",  "ysymbol": "GDX",  "type": "equity"},
    {"code": "SOXX", "ysymbol": "SOXX", "type": "equity"},
    {"code": "BRKB", "ysymbol": "BRK-B","type": "etf"},
    {"code": "XOM",  "ysymbol": "XOM",  "type": "equity"},
    {"code": "HYG",  "ysymbol": "HYG",  "type": "equity"},
    {"code": "DIA",  "ysymbol": "DIA",  "type": "equity"},
    {"code": "XLF",  "ysymbol": "XLF",  "type": "equity"},
    {"code": "XLY",  "ysymbol": "XLY",  "type": "equity"},
    {"code": "XLE",  "ysymbol": "XLE",  "type": "equity"},
    {"code": "XLI",  "ysymbol": "XLI",  "type": "equity"},
    {"code": "KRE",  "ysymbol": "KRE",  "type": "equity"},
    {"code": "XLV",  "ysymbol": "XLV",  "type": "equity"},
    {"code": "SLV",  "ysymbol": "SLV",  "type": "equity"},
    {"code": "EEM",  "ysymbol": "EEM",  "type": "equity"},
    {"code": "XOP",  "ysymbol": "XOP",  "type": "equity"},
    {"code": "XLP",  "ysymbol": "XLP",  "type": "equity"},
    {"code": "XHB",  "ysymbol": "XHB",  "type": "equity"},
    {"code": "VIXY", "ysymbol": "VIXY", "type": "equity"},
    {"code": "EFA",  "ysymbol": "EFA",  "type": "equity"},
    {"code": "XLU",  "ysymbol": "XLU",  "type": "equity"},
    {"code": "VTI",  "ysymbol": "VTI",  "type": "equity"},
    {"code": "USO",  "ysymbol": "USO",  "type": "equity"},  # petróleo
    {"code": "BITO", "ysymbol": "BITO", "type": "equity"},
    {"code": "XLB",  "ysymbol": "XLB",  "type": "equity"},
    {"code": "IYR",  "ysymbol": "IYR",  "type": "equity"},
    {"code": "IBB",  "ysymbol": "IBB",  "type": "equity"},
    {"code": "IEF",  "ysymbol": "IEF",  "type": "equity"},
    {"code": "DBA",  "ysymbol": "DBA",  "type": "equity"},
    {"code": "DBC",  "ysymbol": "DBC",  "type": "equity"},  # commodities (incluye crudo/oro mix)
    {"code": "SHY",  "ysymbol": "SHY",  "type": "equity"},
]

# ============================
# Indicadores técnicos
# ============================

def _ema(a: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(a, dtype=float)
    alpha = 2.0 / (n + 1.0)
    out[:] = np.nan
    if len(a) == 0:
        return out
    out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = alpha * a[i] + (1 - alpha) * out[i - 1]
    return out

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    s = series.astype(float)
    delta = s.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(_ema(up, n), index=s.index)
    roll_down = pd.Series(_ema(down, n), index=s.index)
    rs = roll_up / (roll_down + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev_close = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_close).abs(),
        (l - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    # Cálculo clásico de Wilder
    h = high.astype(float); l = low.astype(float); c = close.astype(float)
    tr = true_range(h, l, c)
    up_move = h.diff()
    down_move = l.diff(-1) * -1  # l.shift(1) - l
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = pd.Series(_ema(tr.values, n), index=tr.index)
    plus_di = 100.0 * pd.Series(_ema(plus_dm, n), index=tr.index) / (atr + 1e-12)
    minus_di = 100.0 * pd.Series(_ema(minus_dm, n), index=tr.index) / (atr + 1e-12)
    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12) ) * 100.0
    adx_ = pd.Series(_ema(dx.values, n), index=tr.index)
    return adx_, plus_di, minus_di

def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, n: int = 14) -> pd.Series:
    tp = (high + low + close) / 3.0
    money_flow = tp * volume
    delta = tp.diff()
    pos_mf = np.where(delta > 0, money_flow, 0.0)
    neg_mf = np.where(delta < 0, money_flow, 0.0)
    pmf = pd.Series(pd.Series(pos_mf).rolling(n).sum().values, index=tp.index)
    nmf = pd.Series(pd.Series(neg_mf).rolling(n).sum().values, index=tp.index)
    mfr = pmf / (nmf + 1e-12)
    return 100.0 - (100.0 / (1.0 + mfr))

def rci(close: pd.Series, n: int = 9) -> pd.Series:
    # Rank Correlation Index
    if len(close) < n:
        return pd.Series(index=close.index, dtype=float)
    rci_vals = []
    idxs = range(len(close))
    for i in idxs:
        if i + 1 < n:
            rci_vals.append(np.nan)
            continue
        window = close.iloc[i + 1 - n : i + 1]
        ranks_time = np.arange(1, n + 1)[::-1]  # t más reciente rank 1
        ranks_price = window.rank(method="first", ascending=True).values
        d = ranks_price - ranks_time
        num = 6.0 * np.sum(d * d)
        den = n * (n + 1) * (2 * n + 1)
        r = 1.0 - (num / den)
        rci_vals.append(r * 100.0)
    return pd.Series(rci_vals, index=close.index)

# ============================
# Métricas de liquidez/volatilidad
# ============================

def _safe_tail(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df.tail(n)

def compute_liquidity_volatility(backend: DataBackend, sym: Dict[str, str]) -> Optional[Dict]:
    """
    Calcula:
      - adv20_usd: promedio de Volumen*Close últimos 20 días
      - atrp: ATR(14) / Close último
      - chop: ATR(5) / (HH(5)-LL(5)) (0..1 aprox, menor es más tendencial)
    """
    symbol = sym["ysymbol"]
    try:
        d1 = backend.history(symbol, "1d")
        d1 = d1.dropna()
        if d1.empty or len(d1) < 40:
            return None
        # ADV USD
        adv20_usd = float((d1["Close"] * d1["Volume"]).rolling(20).mean().iloc[-1])
        # ATR%
        atr14 = true_range(d1["High"], d1["Low"], d1["Close"]).rolling(14).mean()
        atrp = float((atr14.iloc[-1] / (d1["Close"].iloc[-1] + 1e-12)))
        # CHOP (simple)
        hh = d1["High"].rolling(5).max()
        ll = d1["Low"].rolling(5).min()
        rng = (hh - ll).replace(0, np.nan)
        chop = float((atr14.rolling(5).mean().iloc[-1] / (rng.iloc[-1] + 1e-12)))
        # Score básico (liquidez alta + algo de atrp + baja chop)
        score = (
            np.log1p(max(adv20_usd, 1.0)) * 0.6
            + (min(atrp, 0.05) * 100.0) * 0.25
            + ((1.0 - min(chop, 1.0)) * 100.0) * 0.15
        )
        return {
            "code": sym["code"],
            "ysymbol": symbol,
            "type": sym["type"],
            "adv20_usd": adv20_usd,
            "atrp": atrp,
            "chop": chop,
            "score": score,
        }
    except Exception:
        return None

def compute_top50(universe: List[Dict[str,str]]) -> List[Dict]:
    backend = DataBackend()
    rows: List[Dict] = []
    for sym in universe:
        row = compute_liquidity_volatility(backend, sym)
        if row:
            rows.append(row)
    if not rows:
        return []
    # Ordenar por score desc y tomar 50
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:50]

# ============================
# Factores multi-TF para top N
# ============================

def _last_values(series: pd.Series, default: float = np.nan) -> float:
    return float(series.dropna().iloc[-1]) if series is not None and len(series.dropna()) else float('nan')

def _multi_tf_factors(backend: DataBackend, sym: Dict[str,str]) -> Optional[Dict]:
    s = sym["ysymbol"]
    try:
        d15  = backend.history(s, "15m")
        d60  = backend.history(s, "60m")
        d4h  = backend.history(s, "4h")   # del backend ya viene resampleado si es necesario
        d1d  = backend.history(s, "1d")
        if any(df is None or df.empty for df in [d15, d60, d4h, d1d]):
            return None

        # RSI
        rsi15 = _last_values(rsi(d15["Close"], 14))
        rsi60 = _last_values(rsi(d60["Close"], 14))
        rsi4h = _last_values(rsi(d4h["Close"], 14))
        rsi1d = _last_values(rsi(d1d["Close"], 14))

        # ADX (+DI/-DI)
        adx15, pdi15, mdi15 = adx(d15["High"], d15["Low"], d15["Close"], 14)
        adx4h, _, _ = adx(d4h["High"], d4h["Low"], d4h["Close"], 14)

        # MFI y RCI (usar 4H como confirmación medio-plazo)
        mfi4 = _last_values(mfi(d4h["High"], d4h["Low"], d4h["Close"], d4h["Volume"], 14))
        rci4 = _last_values(rci(d4h["Close"], 9))
        # También en 15m para el estado micro
        mfi15 = _last_values(mfi(d15["High"], d15["Low"], d15["Close"], d15["Volume"], 14))
        rci15 = _last_values(rci(d15["Close"], 9))

        # Alineaciones RSI
        rsi_align_long  = (rsi15 >= 50) and (rsi60 >= 55) and (rsi4h >= 55) and (rsi1d >= 55)
        rsi_align_short = (rsi15 <= 50) and (rsi60 <= 45) and (rsi4h <= 45) and (rsi1d <= 45)

        # Señales de fuerza: ADX y +DI / -DI en 15m y ADX en 4H
        adx_ok = (_last_values(adx15) >= 18.0) or (_last_values(adx4h) >= 20.0)

        # Ranking adicional por momentum (aporta ~12 puntos cuando todo alinea)
        rank_bonus = 0.0
        if rsi_align_long:
            rank_bonus += 12.0
        if _last_values(adx15) >= 25.0 or _last_values(adx4h) >= 25.0:
            rank_bonus += 6.0
        if mfi4 >= 60.0 and rci4 >= 60.0:
            rank_bonus += 4.0

        pdi = _last_values(pdi15)
        mdi = _last_values(mdi15)

        return {
            "ysymbol": s,
            "code": sym["code"],
            "type": sym["type"],
            "score50": sym.get("score", 0.0),
            "score12": sym.get("score", 0.0) + rank_bonus,
            "adx": _last_values(adx15),
            "plus_di": pdi,
            "minus_di": mdi,
            "rsi15": rsi15,
            "rsi60": rsi60,
            "rsi4h": rsi4h,
            "rsi1d": rsi1d,
            "rci": rci15,
            "mfi": mfi15,
            "adx4h": _last_values(adx4h),
            "rci4h": rci4,
            "mfi4h": mfi4,
            "rsi_align_long": rsi_align_long,
            "rsi_align_short": rsi_align_short,
            # Estos flags serán refinados por presend_rules:
            "rank_score": sym.get("score", 0.0) + rank_bonus,
            "ob_score": None,
            "ob_pass": True,
            "adx_ok": bool(adx_ok),
            "rsi_align_long_ok": bool(rsi_align_long),
            "rsi_align_short_ok": bool(rsi_align_short),
            "rci_ok": bool(rci4 >= 50.0),
            "mfi_ok": bool(mfi4 >= 50.0),
        }
    except Exception:
        return None

def compute_top3_factors(top50: List[Dict]) -> List[Dict]:
    backend = DataBackend()
    # Toma los mejores 12 por score base para eficiencia
    base = sorted(top50, key=lambda r: r["score"], reverse=True)[:12]
    factors: List[Dict] = []
    for row in base:
        f = _multi_tf_factors(backend, row)
        if f:
            factors.append(f)
    # Ordena por score12 (score base + momentum)
    factors.sort(key=lambda r: r.get("score12", 0.0), reverse=True)
    return factors[:3]

# ============================
# Orquestador
# ============================

def run_full_pipeline() -> Dict:
    """
    Ejecuta el ranking completo y devuelve un payload listo para presend_rules.
    """
    as_of = datetime.now(timezone.utc).date().isoformat()
    top50 = compute_top50(UNIVERSE)
    top3_factors = compute_top3_factors(top50) if top50 else []
    return {
        "as_of": as_of,
        "top50": top50,
        "top3_factors": top3_factors,
    }
