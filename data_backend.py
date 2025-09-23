import datetime as dt
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import yfinance as yf

class DataError(Exception):
    pass

def _canonical_field(col: str) -> Optional[str]:
    s = str(col).strip().lower().replace('_', ' ')
    # señales claras para adj close
    if 'adj close' in s or 'adjusted close' in s or s.endswith('adj close'):
        return 'Adj Close'
    # tomar última palabra como campo si es OHLCV
    last = s.split()[-1]
    if last in ('open','high','low','close','volume'):
        return last.title() if last != 'close' else 'Close'
    # también aceptar casos tipo 'close' en medio (raro pero defensivo)
    if ' close' in s or s.startswith('close'):
        return 'Close'
    return None

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza a columnas: Open, High, Low, Close, Volume (y opcional Adj Close).
    Maneja MultiIndex y columnas duplicadas por ticker colapsándolas
    al vector con más datos válidos (sin NaN).
    """
    if df is None or df.empty:
        raise DataError("Histórico vacío")

    # Si MultiIndex, intentar seleccionar el nivel de campos, manteniendo un solo ticker
    if isinstance(df.columns, pd.MultiIndex):
        # intenta detectar el nivel de campos OHLCV
        fields = {'open','high','low','close','adj close','volume'}
        field_level = None
        for i in range(df.columns.nlevels):
            vals = {str(v).strip().lower() for v in df.columns.get_level_values(i)}
            if any(v in fields for v in vals):
                field_level = i
                break
        if field_level is not None:
            other_levels = [i for i in range(df.columns.nlevels) if i != field_level]
            if other_levels:
                # elige la primera etiqueta de cada otro nivel (primer ticker, etc.)
                slicer = tuple(pd.Index(df.columns.get_level_values(i)).unique()[0] for i in other_levels)
                try:
                    df = df.xs(slicer, axis=1, level=other_levels)
                except Exception:
                    pass
        # si aún queda MultiIndex, aplanar
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [' '.join(map(str, c)).strip() for c in df.columns.to_list()]

    # Si existe columna 'Date', úsala como índice
    if 'Date' in df.columns:
        df = df.set_index('Date')

    # Construye buckets por campo canónico
    buckets: Dict[str, List[pd.Series]] = {'Open':[], 'High':[], 'Low':[], 'Close':[], 'Volume':[], 'Adj Close':[]}
    for col in df.columns:
        f = _canonical_field(col)
        if f is not None and f in buckets:
            s = df[col]
            # Si llega como DataFrame por duplicado de nombre, convierte a serie con la primera columna
            if isinstance(s, pd.DataFrame):
                # elegir la subcolumna con más datos
                counts = s.count()
                best = counts.idxmax()
                s = s[best]
            buckets[f].append(pd.to_numeric(s, errors='coerce'))

    # Selecciona la mejor serie por campo (más valores no nulos)
    out_cols: Dict[str, pd.Series] = {}
    for f, arr in buckets.items():
        if not arr:
            continue
        if len(arr) == 1:
            out_cols[f] = arr[0]
        else:
            counts = [a.count() for a in arr]
            best_idx = int(np.argmax(counts))
            out_cols[f] = arr[best_idx]

    # Si falta Close, pero hay Adj Close, deriva Close
    if 'Close' not in out_cols or out_cols.get('Close') is None:
        if 'Adj Close' in out_cols:
            out_cols['Close'] = out_cols['Adj Close'].copy()
        else:
            # último recurso: buscar cualquier columna original que termine en close
            close_like = [c for c in df.columns if str(c).strip().lower().endswith('close')]
            if close_like:
                out_cols['Close'] = pd.to_numeric(df[close_like[0]], errors='coerce')
            else:
                raise DataError("No hay Close ni Adj Close")

    # Construye el DataFrame final en el orden deseado
    cols_order = ['Open','High','Low','Close','Volume','Adj Close']
    out = pd.DataFrame({k: v for k, v in out_cols.items() if k in cols_order})
    # Limpieza mínima
    for c in ['Open','High','Low','Close']:
        if c not in out.columns:
            raise DataError(f"Falta columna {c}")
    out = out.dropna(subset=['Open','High','Low','Close'])
    if out.shape[0] < 10:
        raise DataError("Muy pocos datos")

    # Índice datetime ordenado
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index, utc=False)
        except Exception:
            pass
    out = out.sort_index()
    return out

class YahooBackend:
    """Backend de datos para ACCIONES/ETFs."""
    def history(self, symbol: str, interval: str, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        df = yf.download(symbol, start=start, end=end, interval=interval,
                         progress=False, auto_adjust=False, group_by=None, threads=False)
        try:
            return _normalize_ohlcv(df)
        except DataError:
            if interval == '60m':
                df2 = yf.download(symbol, start=start, end=end, interval='1h',
                                   progress=False, auto_adjust=False, group_by=None, threads=False)
                return _normalize_ohlcv(df2)
            raise

    def history_bulk(self, symbols: List[str], interval: str, start: dt.datetime, end: dt.datetime) -> Dict[str, pd.DataFrame]:
        syms = list(dict.fromkeys([s for s in symbols if s]))
        out: Dict[str, pd.DataFrame] = {}
        if not syms:
            return out

        data = yf.download(' '.join(syms), start=start, end=end, interval=interval,
                           progress=False, auto_adjust=False, group_by='ticker', threads=True)

        if isinstance(data, pd.DataFrame) and hasattr(data.columns, 'levels'):
            tickers = list(dict.fromkeys(data.columns.get_level_values(0)))
            for tk in syms:
                if tk in tickers:
                    sub = data[tk].copy()
                    try:
                        out[tk] = _normalize_ohlcv(sub)
                    except Exception:
                        pass

        # Completar faltantes individualmente
        missing = [s for s in syms if s not in out]
        for tk in missing:
            try:
                out[tk] = self.history(tk, interval, start, end)
            except DataError:
                if interval == '60m':
                    out[tk] = self.history(tk, '1h', start, end)
                else:
                    continue
        return out

    def order_book(self, symbol: str) -> Optional[dict]:
        return None

def make_backend():
    return YahooBackend()
