
# -*- coding: utf-8 -*-
"""
ranking.py — Pipeline robusto de ranking + señales, con auditoría de datos.
- Señales: ema20>ema50, breakout55
- Score simple: retorno 3m + bonificación por señales
- Logs "quirúrgicos" en español para depurar datos sin ruido
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Dependencia: yfinance (se espera instalada en el entorno de ejecución)
try:
    import yfinance as yf  # type: ignore
except Exception as e:  # pragma: no cover
    yf = None  # Permite importar el módulo aunque falte yfinance
    _import_error = e

# -------------------- Logging --------------------

def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name, None)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}

def _setup_logging() -> logging.Logger:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger("ranking")
    logger.setLevel(level)
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        h.setFormatter(logging.Formatter(fmt))
        logger.addHandler(h)

    # Silenciar librerías verbosas salvo en DEBUG explícito
    debug_3p = _env_bool("DEBUG", False)
    third_party = [
        "yfinance", "urllib3", "peewee",
        "urllib3.connectionpool", "asyncio", "uvicorn"
    ]
    for name in third_party:
        logging.getLogger(name).setLevel(logging.DEBUG if debug_3p else logging.WARNING)

    return logger

log = _setup_logging()

# -------------------- Configuración --------------------

UNIVERSE: List[str] = [
    "SPY","QQQ","IWM","AAPL","NVDA","AMZN",
    "GLD","SLV","SOXX","SMH","GDX","DBC","DBA","XLK","SHY"
]

MIN_ROWS = int(os.getenv("MIN_ROWS", "200"))
MAX_NAN_CLOSE = float(os.getenv("MAX_NAN_CLOSE", "0.01"))  # 1%
FRESH_DAYS_MAX = int(os.getenv("FRESH_DAYS_MAX", "7"))     # datos recientes
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
SLEEP_BASE = float(os.getenv("SLEEP_BASE", "0.8"))

# -------------------- Utilidades --------------------

@dataclass
class TickerAudit:
    ticker: str
    rows: int
    start: Optional[str]
    end: Optional[str]
    nan_close: int
    reason_excluded: Optional[str] = None

def _download_history(ticker: str, range_: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Descarga histórica con reintentos. Devuelve DataFrame con columnas estándar de Yahoo.
    """
    if yf is None:
        raise RuntimeError(f"yfinance no disponible: {_import_error!r}")

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=range_, interval=interval, auto_adjust=False, actions=True)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
            last_exc = RuntimeError("Respuesta vacía de Yahoo")
        except Exception as e:  # pragma: no cover
            last_exc = e
            log.debug("Descarga fallida %s intento %d: %r", ticker, attempt, e)
        # backoff suave
        time.sleep(SLEEP_BASE * attempt)
    assert last_exc is not None
    raise last_exc

def _audit_df(ticker: str, df: Optional[pd.DataFrame]) -> TickerAudit:
    if df is None or df.empty:
        return TickerAudit(ticker, 0, None, None, 0, reason_excluded="vacio")

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    rows = len(df)
    start = df.index.min().date().isoformat() if rows else None
    end = df.index.max().date().isoformat() if rows else None
    nan_close = int(df["Close"].isna().sum()) if "Close" in df.columns else rows

    reason = None
    if rows < MIN_ROWS:
        reason = f"pocos_rows<{MIN_ROWS}"
    elif nan_close / max(rows, 1) > MAX_NAN_CLOSE:
        reason = f"muchos_nan_close>{MAX_NAN_CLOSE:.2%}"
    else:
        # frescura de datos
        last_date = df.index.max().date() if rows else None
        if last_date is not None:
            delta = (pd.Timestamp("today").date() - last_date).days
            if delta > FRESH_DAYS_MAX:
                reason = f"stale>{FRESH_DAYS_MAX}d"

    return TickerAudit(ticker, rows, start, end, nan_close, reason)

def _compute_signals(df: pd.DataFrame) -> Tuple[bool, bool, float]:
    """
    Retorna (ema_cross, breakout55, ret_3m)
    """
    s = df["Close"].astype(float)
    ema20 = s.ewm(span=20, adjust=False).mean()
    ema50 = s.ewm(span=50, adjust=False).mean()
    ema_cross = bool(ema20.iloc[-1] > ema50.iloc[-1])

    # breakout sobre 55 días mirando máximo previo
    rolling_max = s.rolling(window=55, min_periods=55).max()
    # Para evitar marcar breakout si aún no hay 55 períodos
    if np.isnan(rolling_max.iloc[-1]):
        breakout55 = False
    else:
        # breakout si cierre actual > máximo de los últimos 55-1 (excluyendo hoy)
        prev_max = s.iloc[:-1].rolling(window=55, min_periods=55).max().iloc[-1]
        breakout55 = bool(s.iloc[-1] > prev_max if not np.isnan(prev_max) else False)

    # retorno ~3 meses (63 sesiones)
    if len(s) >= 63 and s.iloc[-63] != 0:
        ret_3m = float(s.iloc[-1] / s.iloc[-63] - 1.0)
    else:
        ret_3m = float("nan")

    return ema_cross, breakout55, ret_3m

def _score(ema_cross: bool, breakout55: bool, ret_3m: float) -> float:
    score = 0.0
    if not np.isnan(ret_3m):
        score += ret_3m
    if ema_cross:
        score += 0.5
    if breakout55:
        score += 0.5
    return score

# -------------------- API principal --------------------

def run_full_pipeline(audit: bool = False) -> Dict:
    """
    Ejecuta el pipeline completo y devuelve un payload:
    {
        "as_of": "YYYY-MM-DD",
        "top50": [tickers...],
        "top3_factors": [{"ticker": "...", "reasons": ["ema20>ema50","breakout55"]}, ...]
    }
    """
    as_of = pd.Timestamp("today").date().isoformat()

    # Auditoría de universo
    log.info("DATA_AUDIT | universe | %s", {
        "count": len(UNIVERSE),
        "tickers_sample": UNIVERSE[:10]
    })

    audits: List[TickerAudit] = []
    frames: Dict[str, pd.DataFrame] = {}
    for t in UNIVERSE:
        df = None
        try:
            df = _download_history(t, range_="2y", interval="1d")
        except Exception as e:
            audit = TickerAudit(t, 0, None, None, 0, reason_excluded=f"error_descarga:{type(e).__name__}")
            audits.append(audit)
            if audit and audit.reason_excluded and audit.ticker:
                log.info("DATA_AUDIT | %s", audit.__dict__)
            continue

        # Normalizar columnas clave
        if "Close" not in df.columns:
            audit = TickerAudit(t, len(df), None, None, len(df), reason_excluded="sin_columna_Close")
            audits.append(audit)
            log.info("DATA_AUDIT | %s", audit.__dict__)
            continue

        audit = _audit_df(t, df)
        audits.append(audit)
        log.info("DATA_AUDIT | %s", audit.__dict__)

        if audit.reason_excluded is None:
            frames[t] = df.copy()

    # top50 = válidos por calidad de datos
    valid = list(frames.keys())
    log.info("DATA_AUDIT | %s", {"count": len(valid)})

    # Calcular señales y scores
    scored: List[Tuple[str, float, List[str]]] = []
    top3_factors: List[Dict] = []

    for t in valid:
        df = frames[t]
        try:
            ema_cross, breakout55, ret_3m = _compute_signals(df)
            reasons = []
            if ema_cross: reasons.append("ema20>ema50")
            if breakout55: reasons.append("breakout55")
            sc = _score(ema_cross, breakout55, ret_3m)
            scored.append((t, sc, reasons))
        except Exception as e:
            log.debug("Cálculo de señales falló para %s: %r", t, e)

    # Ordenar por score desc
    scored.sort(key=lambda x: x[1], reverse=True)

    # top3 con razones
    for t, _, reasons in scored[:3]:
        top3_factors.append({"ticker": t, "reasons": reasons})

    # top50: por score si hay, si no, por orden de universo
    if scored:
        top50 = [t for (t, _, _) in scored]
    else:
        top50 = valid

    log.info("DATA_AUDIT | %s", {"count": len(top3_factors), "tickers": [x["ticker"] for x in top3_factors]})
    log.info("RESUMEN | as_of=%s | top50=%d | top3=%d", as_of, len(top50), len(top3_factors))

    payload = {
        "as_of": as_of,
        "top50": top50,
        "top3_factors": top3_factors,
    }

    if audit:
        # Añadimos auditoría resumida al payload (sin ser ruidoso)
        payload["_audit"] = {
            "universe_count": len(UNIVERSE),
            "valid_count": len(valid),
            "excluded": [
                a.__dict__ for a in audits if a.reason_excluded is not None
            ][:20]  # limitar
        }

    return payload

if __name__ == "__main__":  # Ejecución local opcional
    out = run_full_pipeline(audit=True)
    print(out)
