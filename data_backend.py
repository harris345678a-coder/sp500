import datetime as dt
from typing import Optional
import pandas as pd
import numpy as np
import yfinance as yf

class DataError(Exception):
    pass

class YahooBackend:
    """
    Backend estricto: devuelve un DataFrame con columnas estándar
    ['Open','High','Low','Close','Adj Close','Volume'] y al menos 30 filas.
    Si no puede cumplirlo para un símbolo, lanza DataError.
    """
    def __init__(self):
        pass

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            raise DataError("Histórico vacío")

        # Asegurar nombres de columnas con capitalización estándar
        rename_map = {}
        for col in list(df.columns):
            cl = str(col).strip().lower()
            if cl == 'open': rename_map[col] = 'Open'
            elif cl == 'high': rename_map[col] = 'High'
            elif cl == 'low': rename_map[col] = 'Low'
            elif cl in ('close', 'closing'): rename_map[col] = 'Close'
            elif cl in ('adj close', 'adjclose', 'adjusted close'): rename_map[col] = 'Adj Close'
            elif cl == 'volume': rename_map[col] = 'Volume'
        if rename_map:
            df = df.rename(columns=rename_map)

        # Si falta Close pero existe Adj Close, usamos Adj Close (datos reales)
        if 'Close' not in df.columns:
            if 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']
            else:
                raise DataError("No hay columna Close ni Adj Close")

        # Requisitos mínimos
        required = ['Open','High','Low','Close','Volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise DataError(f"Faltan columnas: {missing}")

        # Tipos numéricos y limpieza de NA
        for c in ['Open','High','Low','Close','Adj Close','Volume']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['Open','High','Low','Close'])

        if df.shape[0] < 30:
            raise DataError("Muy pocos datos (mínimo 30 días)")

        return df

    def history(self, ysymbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        # Intento 1: download (más rápido)
        df = yf.download(
            ysymbol, start=start, end=end,
            interval="1d", auto_adjust=False, progress=False, group_by="column", threads=False
        )
        try:
            return self._normalize(df)
        except Exception:
            # Intento 2: auto_adjust True
            df2 = yf.download(
                ysymbol, start=start, end=end,
                interval="1d", auto_adjust=True, progress=False, group_by="column", threads=False
            )
            try:
                return self._normalize(df2)
            except Exception:
                # Intento 3: Ticker().history
                t = yf.Ticker(ysymbol)
                df3 = t.history(start=start, end=end, interval="1d", auto_adjust=False)
                df3 = df3.reset_index().set_index('Date') if 'Date' in df3.columns else df3
                return self._normalize(df3)


def make_backend():
    # En el futuro aquí podrías cambiar a IBKR si DATA_BACKEND=ibkr
    return YahooBackend()
