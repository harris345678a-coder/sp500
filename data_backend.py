from __future__ import annotations
import os
import time
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------- Errors ----------------

class DataError(Exception):
    pass

# ---------------- Utilities ----------------

_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

def _to_naive_utc(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts
    return ts.tz_convert("UTC").tz_localize(None)

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise DataError("DataFrame vacío")
    # yfinance puede traer columnas multi-nivel (cuando se bajan varios tickers)
    if isinstance(df.columns, pd.MultiIndex):
        # Si es un solo símbolo, el primer nivel suele ser el nombre del campo (Open, Close, ...)
        try:
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
        except Exception:
            df = df.droplevel(-1, axis=1)
    # Asegurar columnas esperadas
    for c in _COLS:
        if c not in df.columns:
            df[c] = np.nan
    # Si falta Close, intentar Adj Close
    if df["Close"].isna().all() and not df["Adj Close"].isna().all():
        df["Close"] = df["Adj Close"]
    # A numérico
    for c in _COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Índice datetime y ordenado
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
    df = df.sort_index()
    # Validación final
    if df["Close"].dropna().empty:
        raise DataError("No hay Close ni Adj Close")
    return df[["Open", "High", "Low", "Close", "Volume"]]

def _resample_4h_from_60m(df60: pd.DataFrame) -> pd.DataFrame:
    if df60 is None or df60.empty:
        raise DataError("No hay datos 60m para resample 4H")
    # 4H = 240m => 4 barras de 60m
    o = df60["Open"].resample("4H").first()
    h = df60["High"].resample("4H").max()
    l = df60["Low"].resample("4H").min()
    c = df60["Close"].resample("4H").last()
    v = df60["Volume"].resample("4H").sum(min_count=1)
    out = pd.concat([o, h, l, c, v], axis=1, keys=["Open","High","Low","Close","Volume"]).dropna(how="all")
    out = out.dropna(subset=["Open","High","Low","Close"], how="any")
    if out.empty:
        raise DataError("Resample 4H vacío")
    return out

# ---------------- Backend ----------------

@dataclass
class _CacheItem:
    df: pd.DataFrame
    ts: float  # epoch seconds

class DataBackend:
    """
    Backend de datos con caché TTL corta (por defecto 30s) y soporte 4H.
    Usa yfinance como fuente OHLCV. Todos los cálculos son *en vivo* en cada llamada,
    salvo una caché mínima para no golpear el rate limit.
    """

    def __init__(self, ttl_seconds: int = 30):
        self.ttl = int(ttl_seconds)
        self._cache: Dict[Tuple[str,str], _CacheItem] = {}
        self._lock = threading.Lock()

    # Ventanas por intervalo
    _WINDOW_MAP = {
        "1d": pd.Timedelta(days=365*2),   # 2 años para indicadores diarios
        "60m": pd.Timedelta(days=180),    # 6 meses
        "15m": pd.Timedelta(days=30),     # 1 mes
    }

    def _default_window(self, interval: str) -> pd.Timedelta:
        if interval == "240m":
            return pd.Timedelta(days=90)  # 3 meses (desde 60m)
        return self._WINDOW_MAP.get(interval, pd.Timedelta(days=90))

    def _should_use_cache(self, key: Tuple[str,str]) -> bool:
        item = self._cache.get(key)
        if not item:
            return False
        return (time.time() - item.ts) < self.ttl

    def _put_cache(self, key: Tuple[str,str], df: pd.DataFrame):
        self._cache[key] = _CacheItem(df=df.copy(), ts=time.time())

    def _get_cached(self, key: Tuple[str,str]) -> Optional[pd.DataFrame]:
        item = self._cache.get(key)
        if not item:
            return None
        return item.df.copy()

    def history(self, symbol: str, interval: str, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Devuelve OHLCV para symbol/interval con datos vivos.
        interval soporta: '15m', '60m', '240m', '1d'.
        - 240m se resamplea desde 60m para mayor estabilidad.
        - pre/post: se incluye para intradía (<=60m).
        - TTL de caché configurable (por defecto 30s).
        """
        interval = interval.lower().strip()
        key = (symbol, interval)

        force_env = os.environ.get("FORCE_REFRESH", "0") == "1"
        with self._lock:
            if not force_env and self._should_use_cache(key):
                cached = self._get_cached(key)
                if cached is not None and not cached.empty:
                    return cached

        # Rango temporal
        if end is None:
            end = pd.Timestamp.utcnow().tz_localize("UTC")
        if start is None:
            start = end - self._default_window(interval)

        # Fuente
        if interval == "240m":
            # bajar 60m y resamplear
            base = self.history(symbol, "60m", start, end)
            out = _resample_4h_from_60m(base)
        else:
            yf_interval = interval  # '15m', '60m', '1d' son válidos
            prepost = False if yf_interval == "1d" else True
            # yfinance requiere naive (UTC) para start/end
            s_naive = _to_naive_utc(start)
            e_naive = _to_naive_utc(end)
            df = yf.download(
                symbol,
                start=s_naive,
                end=e_naive,
                interval=yf_interval,
                auto_adjust=False,
                prepost=prepost,
                progress=False,
                threads=False,
            )
            out = _normalize_ohlcv(df)

        with self._lock:
            self._put_cache(key, out)
        return out

    def history_bulk(self, symbols: List[str], interval: str, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for s in symbols:
            try:
                out[s] = self.history(s, interval, start, end)
            except Exception:
                out[s] = pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
        return out
