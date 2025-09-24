# -*- coding: utf-8 -*-
"""
ranking.py
-----------------
Pipeline robusto para construir el universo, auditar datos, calcular factores
y devolver el Top N junto a señales (Top 3) **sin** saltarse filtros obligatorios.

Contrato de importación (app_evolutivo):
    from ranking import run_full_pipeline

>>> Compatibilidad: run_full_pipeline acepta `audit` como kwarg (lo ignora).
    Esto evita errores tipo: TypeError: ... unexpected keyword argument 'audit'

Salida (dict):
{
  "ok": True/False,
  "took_s": float,
  "as_of": "YYYY-MM-DD",
  "top50": [symbol, ...],
  "top3_factors": [{"ticker": str, "reasons": [str, ...]}, ...],
  "diag": {...}  # diagnósticos útiles para logging aguas arriba
}

Notas:
- Universo: ETFs/acciones principales + futuros Oro (GC=F) y Crudo WTI (CL=F). Sin cripto.
- Señales Top3 se eligen **solo** entre candidatos que pasan todos los filtros
  obligatorios (p. ej. tendencia EMA20>EMA50). Si hay <3, se devuelven menos;
  no hay relajación de filtros.
- Logs de auditoría claros al estilo:
    INFO | ranking | DATA_AUDIT | universe | {...}
    INFO | ranking | DATA_AUDIT | {'ticker': 'SPY', 'rows': 501, ...}
    INFO | ranking | RESUMEN | as_of=2025-09-24 | top50=15 | top3=3
- Control de verbosidad por variables de entorno:
    RANKING_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR (default: INFO)
    RANKING_DATA_AUDIT=true|false (default: true)
    YFINANCE_DEBUG=true|false (default: false)
"""

from __future__ import annotations

import os
import time
import math
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise RuntimeError("Pandas es requerido para ranking.py") from e

try:
    import yfinance as yf
except Exception as e:  # pragma: no cover
    raise RuntimeError("yfinance es requerido para ranking.py") from e


__all__ = ["run_full_pipeline", "get_universe"]


# ==============================
# Configuración de logging
# ==============================

_LOGGING_READY = False

def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")

def _setup_logging() -> logging.Logger:
    global _LOGGING_READY
    logger = logging.getLogger("ranking")

    if not _LOGGING_READY:
        level_name = os.getenv("RANKING_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        logger.setLevel(level)

        # Evitar duplicados si el módulo se recarga
        if not logger.handlers:
            handler = logging.StreamHandler()
            fmt = "%(asctime)s | %(levelname)s | ranking | %(message)s"
            handler.setFormatter(logging.Formatter(fmt))
            logger.addHandler(handler)
            logger.propagate = False

        # yfinance logger
        yf_debug = _bool_env("YFINANCE_DEBUG", False)
        yf_logger = logging.getLogger("yfinance")
        yf_logger.setLevel(logging.DEBUG if yf_debug else logging.INFO)
        if yf_debug and not yf_logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | yfinance | %(message)s"))
            yf_logger.addHandler(h)
            yf_logger.propagate = False

        _LOGGING_READY = True

    return logger

log = _setup_logging()
DATA_AUDIT_ON = _bool_env("RANKING_DATA_AUDIT", True)


def _log_audit(context: str, payload: Dict):
    """Formato consistente para auditoría de datos."""
    if DATA_AUDIT_ON and log.isEnabledFor(logging.INFO):
        if context:
            log.info("DATA_AUDIT | %s | %s", context, json.dumps(payload, ensure_ascii=False))
        else:
            log.info("DATA_AUDIT | %s", json.dumps(payload, ensure_ascii=False))


# ==============================
# Universo (sin cripto)
# ==============================

# ETFs/acciones líquidos + Futuros Oro/Crudo (símbolos Yahoo; compatibles con IB)
UNIVERSE_BASE = [
    # ETFs índices/sectores
    "SPY", "QQQ", "IWM", "XLK", "SOXX", "SMH", "GDX", "GLD", "SLV", "DBC", "DBA", "SHY",
    # Mega-caps
    "AAPL", "NVDA", "AMZN",
    # Futuros (Yahoo Finance symbols)
    "GC=F",  # Oro
    "CL=F",  # Crudo WTI
]

# Límite superior para el "Top50": si el universo es menor, se devuelve su tamaño.
TOP50_CAP = 50


# ==============================
# Parámetros/Validadores
# ==============================

@dataclass
class FetchParams:
    period: str = "2y"
    interval: str = "1d"
    auto_adjust: bool = True
    actions: bool = True
    min_rows: int = 200          # mínimo ~1 año hábil
    max_stale_days: int = 7      # última vela no debe estar demasiado desfasada

FETCH = FetchParams()

@dataclass
class Filters:
    # Filtros obligatorios (NO se relajan nunca)
    require_trend_up: bool = True     # EMA20 > EMA50
    min_price: float = 1.0            # evitar penny-stocks
    min_dollar_vol_20d: float = 5e6   # $ volumen promedio 20d mínimo

FILTERS = Filters()


# ==============================
# Utilidades
# ==============================

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _dollar_volume(df: pd.DataFrame) -> pd.Series:
    # Usa 'Close' y 'Volume'; para futuros, Yahoo lo maneja similar.
    if "Close" in df.columns and "Volume" in df.columns:
        return df["Close"] * df["Volume"].fillna(0)
    return pd.Series([0] * len(df), index=df.index)

def _highest(series: pd.Series, lookback: int) -> pd.Series:
    return series.rolling(lookback, min_periods=lookback).max()

def _safe_name(sym: str) -> str:
    # Homogeneiza para logs (GC=F -> GC=F)
    return sym


# ==============================
# Descarga robusta
# ==============================

def _fetch_history(symbol: str, params: FetchParams) -> Optional[pd.DataFrame]:
    """
    Descarga robusta con pequeños reintentos. Devuelve DataFrame con columnas
    al menos ['Open','High','Low','Close','Volume'] o None si falla.
    """
    tries = 3
    backoff = 0.8
    for i in range(1, tries + 1):
        try:
            tkr = yf.Ticker(symbol)
            df = tkr.history(
                period=params.period,
                interval=params.interval,
                auto_adjust=params.auto_adjust,
                actions=params.actions,
            )
            # Normaliza columnas (algunos assets pueden traer 'Adj Close')
            if df is None or df.empty:
                raise RuntimeError("history vacío")
            # Asegura columnas claves
            for col in ("Open", "High", "Low", "Close"):
                if col not in df.columns:
                    # Algunas veces Yahoo nombra "Adj Close"; preferimos "Close"
                    if col == "Close" and "Adj Close" in df.columns:
                        df["Close"] = df["Adj Close"]
                    else:
                        raise RuntimeError(f"columna faltante: {col}")
            return df
        except Exception as e:
            if i == tries:
                log.warning("FETCH_FAIL | %s | intento=%s/%s | %s", symbol, i, tries, repr(e))
                return None
            sleep_s = backoff * i
            log.debug("FETCH_RETRY | %s | intento=%s/%s | durmiendo=%.1fs | %s",
                      symbol, i, tries, sleep_s, repr(e))
            time.sleep(sleep_s)


# ==============================
# Factores y Filtros
# ==============================

from dataclasses import dataclass as _dataclass_reuse

@_dataclass_reuse
class SignalResult:
    ticker: str
    reasons: List[str]

def _apply_mandatory_filters(symbol: str, df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Aplica filtros obligatorios. Si alguno falla, devuelve (False, motivo).
    NO se relajan bajo ninguna circunstancia.
    """
    # Suficientes filas
    if len(df) < FETCH.min_rows:
        return False, f"min_rows<{FETCH.min_rows}"
    # Reciente
    last_dt = df.index[-1]
    if isinstance(last_dt, pd.Timestamp):
        last_dt = last_dt.to_pydatetime()
    if datetime.now(timezone.utc) - last_dt.replace(tzinfo=timezone.utc) > timedelta(days=FETCH.max_stale_days):
        return False, "stale_data"
    # Precio mínimo
    last_close = float(df["Close"].iloc[-1])
    if last_close < FILTERS.min_price:
        return False, f"price<{FILTERS.min_price}"
    # Tendencia EMA20>EMA50 (obligatoria si está activo)
    if FILTERS.require_trend_up:
        ema20 = _ema(df["Close"], 20)
        ema50 = _ema(df["Close"], 50)
        if not bool(ema20.iloc[-1] > ema50.iloc[-1]):
            return False, "ema20<=ema50"
    # Liquidez mínima (usa dollar volume 20d)
    dv = _dollar_volume(df).rolling(20).mean()
    if dv.iloc[-1] < FILTERS.min_dollar_vol_20d:
        return False, f"dollar_vol20<{int(FILTERS.min_dollar_vol_20d)}"
    return True, None

def _compute_signals(symbol: str, df: pd.DataFrame) -> List[str]:
    """
    Cálculo de señales (no obligatorias). Devuelve lista de razones presentes.
    - breakout55: cierre > máximo de los últimos 55 días
    - ema20>ema50: redundante para trazabilidad, ya validado en filtro obligatorio
    - momentum63_pos: rendimiento 63d > 0 (trimestral)
    """
    reasons: List[str] = []
    # Breakout 55
    hi55 = _highest(df["Close"], 55)
    if not math.isnan(hi55.iloc[-1]) and df["Close"].iloc[-1] > hi55.iloc[-1]:
        reasons.append("breakout55")
    # EMA trend
    ema20 = _ema(df["Close"], 20)
    ema50 = _ema(df["Close"], 50)
    if ema20.iloc[-1] > ema50.iloc[-1]:
        reasons.append("ema20>ema50")
    # Momentum 63d
    if len(df) >= 63:
        r63 = df["Close"].pct_change(63).iloc[-1]
        if r63 > 0:
            reasons.append("momentum63_pos")
    return reasons

def _score_for_top50(df: pd.DataFrame) -> float:
    """
    Puntuación simple para ordenar el top50:
      50% momentum 63d + 50% momentum 126d (si disponible).
    """
    close = df["Close"]
    r63 = close.pct_change(63).iloc[-1] if len(close) >= 63 else 0.0
    r126 = close.pct_change(126).iloc[-1] if len(close) >= 126 else r63
    return float(0.5 * r63 + 0.5 * r126)


# ==============================
# Universo dinámico
# ==============================

def get_universe() -> List[str]:
    """
    Devuelve la lista final del universo sin duplicados, con orden estable.
    No incluye cripto. Incluye futuros Oro/Crudo (GC=F, CL=F).
    """
    seen = set()
    ordered = []
    for sym in UNIVERSE_BASE:
        s = _safe_name(sym)
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


# ==============================
# Pipeline principal
# ==============================

def run_full_pipeline(*, audit: Optional[bool] = None, **_) -> Dict:
    """
    Ejecuta el pipeline completo.

    Parámetros
    ----------
    audit : Optional[bool]
        Aceptado por compatibilidad con app_evolutivo; **no modifica** el comportamiento.
    **_ : dict
        Captura kwargs futuros para mantener compatibilidad hacia adelante.
    """
    t0 = time.time()
    as_of = datetime.now().date().isoformat()
    universe = get_universe()

    # Auditoría del universo
    _log_audit("universe", {"count": len(universe), "tickers_sample": universe[:10]})

    fetched: Dict[str, pd.DataFrame] = {}
    excluded: Dict[str, str] = {}
    top50_candidates: List[Tuple[str, float]] = []
    top3: List[SignalResult] = []

    # Descarga y validación
    for sym in universe:
        df = _fetch_history(sym, FETCH)
        if df is None or df.empty:
            excluded[sym] = "fetch_fail_or_empty"
            continue

        # Limpiezas mínimas
        df = df.dropna(subset=["Close"]).copy()

        # Auditoría por símbolo
        payload = {
            "ticker": sym,
            "rows": int(len(df)),
            "start": str(df.index[0].date()) if len(df) else None,
            "end": str(df.index[-1].date()) if len(df) else None,
            "nan_close": int(df["Close"].isna().sum()),
            "reason_excluded": None,
        }

        ok, reason = _apply_mandatory_filters(sym, df)
        if not ok:
            payload["reason_excluded"] = reason
            excluded[sym] = reason or "filter_fail"
        else:
            fetched[sym] = df

        _log_audit("", payload)

    # Ordenamiento por score para "Top50"
    for sym, df in fetched.items():
        score = _score_for_top50(df)
        top50_candidates.append((sym, score))

    top50_candidates.sort(key=lambda x: x[1], reverse=True)
    top50_list = [s for s, _ in top50_candidates[:TOP50_CAP]]

    # Construcción de señales Top3 (NO se relajan filtros)
    # Seleccionamos los 3 mejores por score DENTRO de los que ya pasaron filtros.
    for sym, _ in top50_candidates:
        if len(top3) >= 3:
            break
        df = fetched[sym]
        reasons = _compute_signals(sym, df)
        # Al menos debe mantener la tendencia (ya garantizada) y tener 1 señal útil
        if reasons:
            top3.append(SignalResult(ticker=sym, reasons=reasons))

    # Auditorías finales
    _log_audit("", {"count": len(top50_list)})
    _log_audit("", {"count": len(top3), "tickers": [s.ticker for s in top3]})
    log.info("RESUMEN | as_of=%s | top50=%d | top3=%d", as_of, len(top50_list), len(top3))

    # Diagnóstico transparente
    diag = {
        "universe_count": len(universe),
        "fetched_count": len(fetched),
        "excluded_count": len(excluded),
        "excluded_sample": list({k: v for k, v in list(excluded.items())[:5]}.items()),
    }

    result = {
        "ok": True,
        "took_s": round(time.time() - t0, 2),
        "as_of": as_of,
        "top50": top50_list,
        "top3_factors": [{"ticker": s.ticker, "reasons": s.reasons} for s in top3],
        "diag": diag,
    }
    return result


# Ejecución directa (debug local)
if __name__ == "__main__":
    out = run_full_pipeline()
    print(json.dumps(out, indent=2, ensure_ascii=False))
