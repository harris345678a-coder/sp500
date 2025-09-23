import datetime as dt
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import yfinance as yf

class DataError(Exception):
    pass

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza un DataFrame de yfinance a columnas estándar:
    Open, High, Low, Close, Volume (y opcional Adj Close).
    • Acepta columnas planas o MultiIndex (ticker, campo).
    • Si no hay Close pero hay Adj Close, deriva Close de Adj Close.
    • Si aparece alguna columna que termine en 'close', la usa como Close.
    • Exige como mínimo 10 filas reales.
    """
    if df is None or df.empty:
        raise DataError("Histórico vacío")

    # Si trae MultiIndex (bulk u otras variantes), selecciona la capa de campos
    if isinstance(df.columns, pd.MultiIndex):
        # Detecta el nivel que contiene los campos OHLCV
        field_names = {"open", "high", "low", "close", "adj close", "volume"}
        field_level = None
        for i in range(df.columns.nlevels):
            vals = {str(v).strip().lower() for v in df.columns.get_level_values(i)}
            if any(v in field_names for v in vals):
                field_level = i
                break
        # Si existe field_level, colapsa los otros niveles eligiendo el primer valor
        if field_level is not None:
            other_levels = [i for i in range(df.columns.nlevels) if i != field_level]
            # Si hay múltiples etiquetas en otros niveles, se toma la primera combinación
            if other_levels:
                # Construye slicer tomando el primer valor único de cada otro nivel
                slicer = tuple(pd.Index(df.columns.get_level_values(i)).unique()[0] for i in other_levels)
                df = df.xs(slicer, axis=1, level=other_levels)
        # Si aún queda multiindex o no detectó nivel de campos, aplanar a nombres "AAPL Close"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(map(str, c)).strip() for c in df.columns.to_list()]

    # Si existe columna 'Date', úsala como índice temporal
    if 'Date' in df.columns:
        df = df.set_index('Date')

    # Renombrado canónico
    rename_map = {}
    for col in list(df.columns):
        cl = str(col).strip().lower()
        # Casos "AAPL Close" -> toma la última palabra como campo
        if ' ' in cl and cl.split()[-1] in ('open','high','low','close','volume'):
            cl = cl.split()[-1]
        if cl == 'open': rename_map[col] = 'Open'
        elif cl == 'high': rename_map[col] = 'High'
        elif cl == 'low': rename_map[col] = 'Low'
        elif cl in ('close','closing'): rename_map[col] = 'Close'
        elif cl in ('adj close','adjclose','adjusted close'): rename_map[col] = 'Adj Close'
        elif cl == 'volume': rename_map[col] = 'Volume'
    if rename_map:
        df = df.rename(columns=rename_map)

    # Deriva Close si falta
    if 'Close' not in df.columns:
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        else:
            # Último recurso: columna que termine con 'close'
            maybe = [c for c in df.columns if str(c).strip().lower().endswith('close')]
            if maybe:
                df['Close'] = pd.to_numeric(df[maybe[0]], errors='coerce')
            else:
                raise DataError("No hay Close ni Adj Close")

    # Tipos numéricos y limpieza mínima
    for c in ['Open','High','Low','Close','Adj Close','Volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['Open','High','Low','Close'])
    if df.shape[0] < 10:
        raise DataError("Muy pocos datos")

    # Asegura índice datetime ordenado
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, utc=False)
        except Exception:
            pass
    df = df.sort_index()
    return df

class YahooBackend:
    """
    Backend de datos para ACCIONES/ETFs (no cripto).
    - history(symbol, interval, start, end) -> DataFrame con columnas: Open, High, Low, Close, Volume
    - history_bulk(symbols, interval, start, end) -> dict[symbol] = DataFrame
    - order_book(symbol) -> None (Yahoo no provee Book). IBKR lo cubriría.
    """
    def history(self, symbol: str, interval: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        # Intento principal
        df = yf.download(symbol, start=start, end=end, interval=interval,
                         progress=False, auto_adjust=False, group_by=None, threads=False)
        try:
            return _normalize_ohlcv(df)
        except DataError:
            # Fallback quirúrgico: 60m -> 1h (naming alterno de yfinance)
            if interval == '60m':
                df2 = yf.download(symbol, start=start, end=end, interval='1h',
                                   progress=False, auto_adjust=False, group_by=None, threads=False)
                return _normalize_ohlcv(df2)
            # Si vino con MultiIndex raro, intenta aplanado genérico (manejado adentro)
            # y si igual falla, re-levanta el error sin más "parches"
            raise

    def history_bulk(self, symbols: List[str], interval: str, start: dt.datetime, end: dt.datetime) -> Dict[str, pd.DataFrame]:
        # Descarga en bloque; si falla un ticker, se completa con history() puntual del mismo intervalo
        syms = list(dict.fromkeys([s for s in symbols if s]))
        out: Dict[str, pd.DataFrame] = {}
        if not syms:
            return out

        data = yf.download(' '.join(syms), start=start, end=end, interval=interval,
                           progress=False, auto_adjust=False, group_by='ticker', threads=True)

        # Caso típico: MultiIndex (ticker, field)
        if isinstance(data, pd.DataFrame) and hasattr(data.columns, 'levels'):
            tickers = list(dict.fromkeys(data.columns.get_level_values(0)))
            for tk in syms:
                if tk in tickers:
                    sub = data[tk].copy()
                    try:
                        out[tk] = _normalize_ohlcv(sub)
                    except Exception:
                        pass

        # Completa los que no salieron del bulk con pedidos unitarios
        missing = [s for s in syms if s not in out]
        for tk in missing:
            try:
                out[tk] = self.history(tk, interval, start, end)
            except DataError:
                if interval == '60m':
                    # ÚNICO fallback permitido: 60m -> 1h
                    out[tk] = self.history(tk, '1h', start, end)
                else:
                    # Si tampoco sale, se omite ese ticker sin bloquear todo el job
                    continue
        return out

    def order_book(self, symbol: str) -> Optional[dict]:
        # Yahoo no provee L2; esta interfaz existe para IBKR en el futuro.
        return None

def make_backend():
    return YahooBackend()
