import os
import datetime as dt
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import yfinance as yf

class DataError(Exception):
    pass

class YahooBackend:
    """
    Backend de datos para ACCIONES/ETFs (no cripto).
    - history(symbol, interval, start, end) -> DataFrame con columnas: Open, High, Low, Close, Volume
    - history_bulk(symbols, interval, start, end) -> dict[symbol] = DataFrame
    - order_book(symbol) -> None (Yahoo no provee Book). IBKR lo cubriría.
    """
    def __init__(self):
        pass

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            raise DataError("Histórico vacío")
        rename_map = {}
        for col in list(df.columns):
            cl = str(col).strip().lower()
            if cl == 'open': rename_map[col] = 'Open'
            elif cl == 'high': rename_map[col] = 'High'
            elif cl == 'low': rename_map[col] = 'Low'
            elif cl in ('close','closing'): rename_map[col] = 'Close'
            elif cl in ('adj close','adjclose','adjusted close'): rename_map[col] = 'Adj Close'
            elif cl == 'volume': rename_map[col] = 'Volume'
        if rename_map:
            df = df.rename(columns=rename_map)
        if 'Close' not in df.columns:
            if 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']
            else:
                raise DataError("No hay Close ni Adj Close")
        for c in ['Open','High','Low','Close','Adj Close','Volume']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['Open','High','Low','Close'])
        if df.shape[0] < 10:
            raise DataError("Muy pocos datos")
        return df

    def history(self, symbol: str, interval: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        df = yf.download(symbol, start=start, end=end, interval=interval, progress=False, auto_adjust=False, group_by=None, threads=False)
        if isinstance(df, pd.DataFrame) and not df.empty and 'Date' in df.columns:
            df = df.set_index('Date')
        return self._normalize(df)

    def history_bulk(self, symbols: List[str], interval: str, start: dt.datetime, end: dt.datetime) -> Dict[str, pd.DataFrame]:
        syms = list(dict.fromkeys([s for s in symbols if s]))
        if not syms:
            return {}
        data = yf.download(' '.join(syms), start=start, end=end, interval=interval,
                           progress=False, auto_adjust=False, group_by='ticker', threads=True)
        out = {}
        if isinstance(data, pd.DataFrame) and hasattr(data.columns, 'levels'):
            tickers = list(dict.fromkeys(data.columns.get_level_values(0)))
            for tk in syms:
                if tk in tickers:
                    sub = data[tk].copy()
                    try:
                        out[tk] = self._normalize(sub)
                    except Exception:
                        pass
        missing = [s for s in syms if s not in out]
        for tk in missing:
            try:
                out[tk] = self.history(tk, interval, start, end)
            except Exception:
                pass
        return out

    def order_book(self, symbol: str) -> Optional[dict]:
        return None

def make_backend():
    return YahooBackend()
