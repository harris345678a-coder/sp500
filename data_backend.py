import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional

class DataError(Exception):
    pass

_PERIOD_BY_INTERVAL = {
    "1d": "400d",     # enough for ADV20 and EMAs
    "60m": "90d",
    "15m": "30d",     # Yahoo caps shorter intervals
}

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [" ".join(map(str, c)).strip() for c in df.columns]
    return df

def _canonical(col: str) -> Optional[str]:
    s = str(col).strip().lower()
    if s.endswith(" open") or s == "open":
        return "Open"
    if s.endswith(" high") or s == "high":
        return "High"
    if s.endswith(" low") or s == "low":
        return "Low"
    if s.endswith(" close") or s == "close":
        return "Close"
    if s.endswith(" adj close") or s == "adj close" or "adjusted close" in s:
        return "Adj Close"
    if s.endswith(" volume") or s == "volume":
        return "Volume"
    return None

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = _flatten(df)
    buckets: Dict[str, list] = {"Open":[], "High":[], "Low":[], "Close":[], "Adj Close":[], "Volume":[]}

    for col in df.columns:
        canon = _canonical(col)
        if canon is None:
            continue
        s = df[col]
        if isinstance(s, pd.DataFrame):
            for c in s.columns:
                buckets[canon].append(pd.to_numeric(s[c], errors="coerce"))
        else:
            buckets[canon].append(pd.to_numeric(s, errors="coerce"))

    out: Dict[str, pd.Series] = {}
    for key in ("Open","High","Low","Close","Adj Close","Volume"):
        if buckets[key]:
            counts = [ser.notna().sum() for ser in buckets[key]]
            best = int(np.argmax(counts))
            out[key] = buckets[key][best]

    # Fallback Close -> Adj Close
    if "Close" not in out or out["Close"].isna().all():
        if "Adj Close" in out and not out["Adj Close"].isna().all():
            out["Close"] = out["Adj Close"]
        else:
            return pd.DataFrame()

    res = pd.DataFrame({k: v for k, v in out.items() if k in ("Open","High","Low","Close","Volume")})
    res = res.dropna(subset=["Open","High","Low","Close"])
    if res.empty:
        return res
    if not isinstance(res.index, pd.DatetimeIndex):
        try:
            res.index = pd.to_datetime(res.index, utc=False)
        except Exception:
            pass
    return res.sort_index()

class DataBackend:
    """Yahoo Finance backend with robust normalization and 4H via resampling 60m."""
    def __init__(self) -> None:
        pass

    def history(self, symbol: str, interval: str, start=None, end=None) -> pd.DataFrame:
        if interval not in ("15m","60m","240m","1d"):
            raise DataError(f"Intervalo no soportado: {interval}")
        base_interval = "60m" if interval == "240m" else interval
        period = _PERIOD_BY_INTERVAL.get(base_interval, "90d")
        try:
            df = yf.download(symbol, period=period, interval=base_interval, progress=False, auto_adjust=False, group_by=None, threads=False)
        except Exception as e:
            raise DataError(f"Descarga falló para {symbol} [{base_interval}]: {e}")

        nd = _normalize_ohlcv(df)
        if nd.empty:
            return nd

        if interval == "240m":
            # Resample 60m -> 4H
            agg = {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
            nd = nd.resample("4H", label="right", closed="right").agg(agg).dropna()

        return nd

    def history_bulk(self, symbols: List[str], interval: str, start=None, end=None) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for tk in symbols:
            try:
                out[tk] = self.history(tk, interval, start, end)
            except Exception:
                out[tk] = pd.DataFrame()
        return out
