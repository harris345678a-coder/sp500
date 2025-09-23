
import os
import time
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ================================
# Time helpers (robust, UTC-only)
# ================================

def utc_now() -> pd.Timestamp:
    """
    Return current time as timezone-aware pandas Timestamp in UTC.
    Never tz_localize on already-aware objects.
    """
    return pd.Timestamp.now(tz="UTC")


def ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tz is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


# ================================
# Normalization helpers
# ================================

_OHLCV_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise DataError("DataFrame vacío de proveedor")

    # Standardize columns (yfinance returns these exact names usually)
    for c in _OHLCV_COLS:
        if c not in df.columns:
            # allow absence of Adj Close; others must exist
            if c == "Adj Close":
                continue
            # try alternate naming
            alt = c.replace(" ", "")
            if alt in df.columns:
                df[c] = df[alt]
            else:
                # If critical column missing, fail explicitly
                raise DataError(f"Falta columna requerida: {c}")

    # Coerce numeric safely
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # If Close has many NaNs, try Adj Close as fallback
    if df["Close"].isna().all():
        if "Adj Close" in df and not df["Adj Close"].isna().all():
            df["Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")
        else:
            raise DataError("No hay Close ni Adj Close")

    # Index must be UTC-aware DateTimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Datetime" in df.columns:
            df = df.set_index(pd.to_datetime(df["Datetime"], utc=True))
        else:
            raise DataError("Índice sin fechas válidas")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Drop rows without Close (still)
    df = df.dropna(subset=["Close"])
    if len(df) == 0:
        raise DataError("Todos los Close son NaN tras normalización")

    # Sort ascending and ensure unique index
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _resample_to_4h_from_60m(df_60m: pd.DataFrame) -> pd.DataFrame:
    """
    Create a stable 4-hour OHLCV resample from 60m bars.
    Uses lower-case 'h' to avoid pandas deprecation warning.
    Bars are right-labeled/closed to align with trading platforms.
    """
    if df_60m is None or len(df_60m) == 0:
        raise DataError("No hay datos 60m para construir 4h")
    need_cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in need_cols:
        if c not in df_60m.columns:
            raise DataError(f"Falta {c} en 60m para 4h")

    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    nd = df_60m.resample("4h", label="right", closed="right").agg(agg).dropna()
    # Asegurar índice UTC y sin duplicados
    if nd.index.tz is None:
        nd.index = nd.index.tz_localize("UTC")
    else:
        nd.index = nd.index.tz_convert("UTC")
    nd = nd[~nd.index.duplicated(keep="last")]
    return nd


# ================================
# Caching (simple TTL to avoid rate limits)
# ================================

class _TTLCache:
    def __init__(self, ttl_seconds: int = 30) -> None:
        self.ttl = ttl_seconds
        self.store: Dict[Tuple[str, str], Tuple[float, pd.DataFrame]] = {}

    def get(self, key: Tuple[str, str]) -> Optional[pd.DataFrame]:
        ts = time.time()
        if key in self.store:
            t0, df = self.store[key]
            if ts - t0 <= self.ttl:
                return df.copy()
            else:
                self.store.pop(key, None)
        return None

    def put(self, key: Tuple[str, str], df: pd.DataFrame) -> None:
        self.store[key] = (time.time(), df.copy())


# ================================
# Public API
# ================================

class DataError(RuntimeError):
    pass


class YFBackend:
    """
    Backend de datos usando yfinance, con normalización robusta y soporte 15m/60m/240m/1d.
    """

    def __init__(self) -> None:
        ttl = 0 if os.environ.get("FORCE_REFRESH") == "1" else 30
        self.cache = _TTLCache(ttl_seconds=ttl)

    def _download(self, symbol: str, interval: str, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
        # yfinance espera tz-aware o naive; pasamos tz-aware UTC siempre.
        start = ensure_utc(start) if start is not None else None
        end = ensure_utc(end) if end is not None else utc_now()

        prepost = interval.lower().endswith("m")
        # yfinance param interval admite: "1m","5m","15m","60m","90m","1h","1d", etc.
        # Usamos "60m" para 60 y construiremos 4h por resample si piden 240m.
        actual_interval = "60m" if interval == "240m" else interval

        df = yf.download(
            tickers=symbol,
            interval=actual_interval,
            start=None if start is None else start.tz_convert("UTC").to_pydatetime(),
            end=None if end is None else end.tz_convert("UTC").to_pydatetime(),
            prepost=prepost,
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if isinstance(df, pd.DataFrame) and "Ticker" in df.columns:
            # yfinance multi-ticker form -> single
            if symbol in df.columns.get_level_values(1):
                df = df.xs(symbol, axis=1, level=1)
        return df

    def history(self, symbol: str, interval: str, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Retorna OHLCV normalizado (UTC), soporta: 15m, 60m, 240m (4h), 1d
        """
        key = (symbol, interval)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        df_raw = self._download(symbol, interval, start, end)
        if df_raw is None or len(df_raw) == 0:
            raise DataError(f"Proveedor vacío: {symbol} {interval}")

        if interval == "240m":
            # construir 4h desde 60m para estabilidad
            base = _normalize_ohlcv(df_raw)
            df = _resample_to_4h_from_60m(base)
        else:
            df = _normalize_ohlcv(df_raw)

        self.cache.put(key, df)
        return df

    def history_bulk(self, symbols: Iterable[str], interval: str, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for tk in symbols:
            try:
                out[tk] = self.history(tk, interval, start, end)
            except Exception as e:
                # Propagamos error detallado por símbolo; el pipeline superior decide
                raise DataError(f"history_bulk fallo para {tk} @ {interval}: {e}") from e
        return out
