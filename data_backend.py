
# data_backend.py
# Backend de datos robusto para Yahoo Finance (yfinance)
# - Soporta intervalos: 15m, 60m, 240m (4H), 1d
# - Resample 4H desde 60m (sin warnings)
# - Normalización OHLCV (maneja "Adj Close")
# - Índices UTC, ordenados, sin duplicados
# - Reintentos con 2 métodos (Ticker.history y yf.download)
# - Excepciones claras (DataError) para que el pipeline pueda continuar

from __future__ import annotations

import time
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yfinance as yf


class DataError(Exception):
    pass


_SUPPORTED = {"15m", "60m", "240m", "1d"}

# Límites de Yahoo por intervalo
_PERIOD_BY_INTERVAL = {
    "15m": "60d",   # máx para 15m
    "60m": "730d",  # 2 años en 60m es aceptado
    "1d":  "2y",    # suficiente para indicadores
}


def _utc_now() -> pd.Timestamp:
    # seguro tz-aware
    return pd.Timestamp.now(tz="UTC")


class DataBackend:
    def __init__(self, auto_adjust: bool = False, max_retries: int = 2, retry_sleep: float = 0.8):
        self.auto_adjust = auto_adjust
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep

    # API principal
    def history(
        self,
        symbol: str,
        interval: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        interval = interval.lower()
        if interval not in _SUPPORTED:
            raise DataError(f"Intervalo no soportado: {interval}")

        # 4H se construye desde 60m
        if interval == "240m":
            base = self.history(symbol, "60m", start=start, end=end)
            if base.empty:
                raise DataError(f"Sin datos base 60m para {symbol} (armar 240m)")
            # Resample a 4 horas (label y closed a la derecha)
            agg = {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
            # índice debe estar en UTC y ser DatetimeIndex; _normalize_ohlcv ya lo garantiza
            out = (
                base.resample("4h", label="right", closed="right")
                    .agg(agg)
                    .dropna()
            )
            if out.empty:
                raise DataError(f"Resample 4h vacío para {symbol}")
            return out

        # Para 1d/60m/15m traemos directo
        df = self._fetch_ohlcv(symbol, interval, start, end)
        return self._normalize_ohlcv(df, symbol, interval)

    def history_bulk(
        self,
        symbols: Iterable[str],
        interval: str,
        start: Optional[pd.Timestamp] = None,
        end: Optional[pd.Timestamp] = None,
    ) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for s in symbols:
            try:
                out[s] = self.history(s, interval, start, end)
            except DataError:
                # se omite el símbolo problemático; el pipeline debe continuar
                continue
        if not out:
            raise DataError(f"No se pudo obtener ningún dataset para interval={interval}")
        return out

    # ----------------- Internals -----------------

    def _fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> pd.DataFrame:
        """Obtiene el DataFrame bruto desde Yahoo con reintentos y 2 métodos."""
        base_interval = "60m" if interval == "240m" else interval

        # Normaliza tiempos (tz-aware -> UTC); si vienen naive, localiza en UTC
        def _to_utc(ts: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
            if ts is None:
                return None
            if ts.tz is None:
                return ts.tz_localize("UTC")
            return ts.tz_convert("UTC")

        start_utc = _to_utc(start)
        end_utc = _to_utc(end) or _utc_now()

        # Si no se pasó rango, usamos "period" recomendado por Yahoo
        period = None
        if start_utc is None and end_utc is not None:
            period = _PERIOD_BY_INTERVAL.get(base_interval, "2y")

        # Dos estrategias: Ticker.history y yf.download
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                tkr = yf.Ticker(symbol)
                if period:
                    df = tkr.history(
                        period=period,
                        interval=base_interval,
                        auto_adjust=self.auto_adjust,
                        prepost=False,
                        actions=False,
                    )
                else:
                    df = tkr.history(
                        start=start_utc,
                        end=end_utc,
                        interval=base_interval,
                        auto_adjust=self.auto_adjust,
                        prepost=False,
                        actions=False,
                    )

                if self._looks_valid(df):
                    return df

                # Fallback 1: yf.download (suele comportarse distinto)
                if period:
                    df2 = yf.download(
                        tickers=symbol,
                        period=period,
                        interval=base_interval,
                        auto_adjust=self.auto_adjust,
                        prepost=False,
                        progress=False,
                        group_by="column",
                        threads=False,
                    )
                else:
                    df2 = yf.download(
                        tickers=symbol,
                        start=start_utc,
                        end=end_utc,
                        interval=base_interval,
                        auto_adjust=self.auto_adjust,
                        prepost=False,
                        progress=False,
                        group_by="column",
                        threads=False,
                    )

                if self._looks_valid(df2):
                    return df2

                last_err = DataError(f"Respuesta vacía/ inválida para {symbol} @ {base_interval}")
            except Exception as e:
                last_err = e

            if attempt < self.max_retries:
                time.sleep(self.retry_sleep)

        raise DataError(f"No se pudo descargar datos para {symbol} @ {base_interval}: {last_err}")

    @staticmethod
    def _looks_valid(df: Optional[pd.DataFrame]) -> bool:
        if df is None or len(df) == 0:
            return False
        # Algunas veces Yahoo devuelve columnas en minúsculas o solo 'Adj Close'
        cols = {c.lower() for c in df.columns}
        needed_any = {"close"} | {"adj close"}
        return any(k in cols for k in needed_any)

    @staticmethod
    def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        # Flatten si viene MultiIndex (group_by="ticker")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [" ".join([str(x) for x in tup if str(x) != ""]).strip() for tup in df.columns]

        # Renombrado tolerante
        rename_map = {}
        for c in df.columns:
            cl = c.strip().lower().replace("_", " ")
            if cl in ("open",):
                rename_map[c] = "Open"
            elif cl in ("high",):
                rename_map[c] = "High"
            elif cl in ("low",):
                rename_map[c] = "Low"
            elif cl in ("close", "closing price"):
                rename_map[c] = "Close"
            elif cl in ("adj close", "adjusted close", "adjacent close"):
                rename_map[c] = "Adj Close"
            elif cl in ("volume",):
                rename_map[c] = "Volume"
        df = df.rename(columns=rename_map)

        # Si no hay Close pero sí Adj Close, lo usamos como Close (dato real ajustado)
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]

        return df

    def _normalize_ohlcv(self, df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
        if df is None or df.empty:
            raise DataError(f"Sin datos para {symbol} en {interval}")

        df = self._standardize_columns(df)

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise DataError(f"Columnas faltantes {missing} para {symbol} en {interval}")

        # Índice a DateTimeIndex y UTC
        if not isinstance(df.index, pd.DatetimeIndex):
            # intenta columnas comunes de fecha
            for dc in ("Datetime", "Date", "date", "timestamp"):
                if dc in df.columns:
                    df = df.set_index(pd.to_datetime(df[dc], utc=True))
                    break
        if not isinstance(df.index, pd.DatetimeIndex):
            raise DataError(f"Índice no temporal para {symbol} en {interval}")

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        # Mantener solo columnas requeridas
        df = df[required].copy()

        # Tipos numéricos limpios
        for c in required:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Elimina filas sin Close o con NaN críticos
        df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
        if df.empty:
            raise DataError(f"Todos los registros inválidos para {symbol} en {interval}")

        # Volume NaN -> 0 (Yahoo a veces devuelve NaN intradía)
        if df["Volume"].isna().any():
            df["Volume"] = df["Volume"].fillna(0)

        # Orden y duplicados
        df = df[~df.index.duplicated(keep="last")].sort_index()

        return df
