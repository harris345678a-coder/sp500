
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional

__all__ = ["DataBackend", "DataError"]


class DataError(Exception):
    pass


def _now_utc() -> pd.Timestamp:
    """Return a timezone-aware UTC timestamp."""
    return pd.Timestamp.now(tz="UTC")


def _ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    if ts is None:
        return _now_utc()
    if isinstance(ts, pd.Timestamp):
        if ts.tz is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")
    # Try to parse anything else
    t = pd.Timestamp(ts)
    if t.tz is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def _to_naive_utc(ts: pd.Timestamp):
    uts = _ensure_utc(ts)
    return uts.tz_convert("UTC").tz_localize(None)


class DataBackend:
    """Yahoo-backed OHLCV provider with normalization and 4h resampling.

    Supports intervals: 15m, 60m, 240m (built from 60m), 1d.
    Always returns a pandas.DataFrame indexed by UTC timestamps with columns:
    Open, High, Low, Close, Volume (and Adj Close when available).
    """

    def __init__(self, cache_ttl_seconds: int = 600):
        self.cache_ttl = pd.Timedelta(seconds=cache_ttl_seconds)
        # cache: key -> (df, fetched_at_utc)
        self._cache: Dict[tuple, tuple] = {}

    # ---------- public API ----------
    def history(
        self,
        symbol: str,
        interval: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        interval = interval.lower()
        if interval not in {"15m", "60m", "240m", "1d"}:
            raise DataError(f"Intervalo no soportado: {interval}")

        end = _ensure_utc(end or _now_utc())
        start = _ensure_utc(start or self._default_start(interval, end))

        if interval == "240m":
            # Construimos 4h desde 60m
            base_df = self._download(symbol, "60m", start, end)
            if base_df.empty:
                raise DataError(f"Sin datos 60m para construir 4h en {symbol}")
            df = self._resample_4h_from_60m(base_df)
        else:
            df = self._download(symbol, interval, start, end)

        return self._normalize_ohlcv(df, symbol, interval)

    def history_bulk(
        self,
        symbols: List[str],
        interval: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                out[sym] = self.history(sym, interval, start=start, end=end)
            except DataError:
                # Saltamos símbolos sin datos válidos, devolvemos el resto
                continue
        return out

    # ---------- internals ----------
    def _default_start(self, interval: str, end: pd.Timestamp) -> pd.Timestamp:
        if interval == "1d":
            delta = pd.Timedelta(days=500)
        elif interval in {"60m", "240m"}:
            # Límite intradía de Yahoo suele rondar 60-90 días
            delta = pd.Timedelta(days=70)
        elif interval == "15m":
            delta = pd.Timedelta(days=35)
        else:
            delta = pd.Timedelta(days=60)
        return end - delta

    def _cache_key(self, symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp):
        # Redondeamos tiempos a minuto para mejorar hit-rate
        s = _ensure_utc(start).floor("min")
        e = _ensure_utc(end).ceil("min")
        return (symbol.upper(), interval, s.value, e.value)

    def _download(self, symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        key = self._cache_key(symbol, interval, start, end)
        cached = self._cache.get(key)
        now = _now_utc()
        if cached is not None:
            df, fetched_at = cached
            if now - fetched_at < self.cache_ttl:
                return df.copy()

        yf_interval = interval if interval in {"1d", "60m", "15m"} else "60m"  # 240m usa 60m base
        try:
            df = yf.download(
                tickers=symbol,
                interval=yf_interval,
                start=_to_naive_utc(start),
                end=_to_naive_utc(end),
                auto_adjust=False,
                progress=False,
                prepost=False,
                threads=False,
            )
        except Exception as e:
            raise DataError(f"Fallo descarga Yahoo para {symbol} {interval}: {e}")

        # yfinance puede devolver columnas en MultiIndex si hay varios tickers
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(0, axis=1)

        # Asegurar índice UTC
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataError(f"Índice no temporal en {symbol} {interval}")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df = df.sort_index().copy()
        # Guardar cache
        self._cache[key] = (df.copy(), now)
        return df

    def _resample_4h_from_60m(self, df60: pd.DataFrame) -> pd.DataFrame:
        # Asegurar columnas básicas antes del resample
        needed = ["Open", "High", "Low", "Close", "Volume"]
        cols_lower = {c.lower(): c for c in df60.columns}
        # Subsanar mayúsculas/minúsculas
        for n in list(df60.columns):
            if n.lower() in {"open", "high", "low", "close", "adj close", "volume"} and n not in needed and n.title() in needed:
                df60.rename(columns={n: n.title()}, inplace=True)

        # Si falta Close pero hay Adj Close, lo usamos
        if "Close" not in df60.columns and "Adj Close" in df60.columns:
            df60["Close"] = df60["Adj Close"]

        agg = {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
        # Adj Close si existe, la tomamos como last
        if "Adj Close" in df60.columns:
            agg["Adj Close"] = "last"

        # Nota: usar '4h' (lowercase) para evitar FutureWarning
        df4h = df60.resample("4h", label="right", closed="right").agg(agg)

        # Limpiar huecos
        df4h = df4h.dropna(how="all")
        # Si Close queda NaN en velas, quitarlas
        if "Close" in df4h.columns:
            df4h = df4h.dropna(subset=["Close"])

        return df4h

    def _normalize_ohlcv(self, df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
        if df is None or len(df) == 0:
            raise DataError(f"Datos vacíos para {symbol} en {interval}")

        # Normalizar nombres
        rename_map = {}
        for c in df.columns:
            lc = c.lower()
            if lc == "adj close":
                rename_map[c] = "Adj Close"
            elif lc in {"open", "high", "low", "close", "volume"}:
                rename_map[c] = lc.title()
        if rename_map:
            df = df.rename(columns=rename_map)

        # Si no hay Close pero sí Adj Close, úsala
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]

        # Asegurar columnas mínimas
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise DataError(f"Columnas faltantes {missing} para {symbol} en {interval}")

        # Coerción numérica robusta
        for c in required + (["Adj Close"] if "Adj Close" in df.columns else []):
            if c in df.columns:
                # si es Series-like, convertimos; si no, error
                ser = df[c]
                if not isinstance(ser, (pd.Series, pd.core.series.Series)):
                    raise DataError(f"Columna {c} inválida para {symbol} en {interval}")
                df[c] = pd.to_numeric(ser, errors="coerce")

        # Índice UTC, único y ordenado
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataError(f"Índice no temporal en {symbol} {interval}")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df[~df.index.duplicated(keep="last")].sort_index()

        # Remover filas sin Close
        df = df.dropna(subset=["Close"])
        if df.empty:
            raise DataError(f"Después de normalizar no quedó Close para {symbol} en {interval}")

        return df
