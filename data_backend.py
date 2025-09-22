import os
import datetime as dt
from typing import Dict, List, Optional, Tuple
import pandas as pd

class OHLCVBackend:
    def history(self, ysymbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Devuelve DataFrame con columnas: ['Open','High','Low','Close','Adj Close','Volume'] índice datetime."""
        raise NotImplementedError

class YahooBackend(OHLCVBackend):
    def __init__(self):
        try:
            import yfinance as yf  # lazy import
            self.yf = yf
        except Exception as e:
            raise RuntimeError("Para usar el backend 'yahoo' debes instalar yfinance") from e

    def history(self, ysymbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        df = self.yf.download(ysymbol, start=start, end=end, progress=False, auto_adjust=False, threads=False)
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame(columns=['Open','High','Low','Close','Adj Close','Volume'])
        return df[['Open','High','Low','Close','Adj Close','Volume']]

class IBKRBackend(OHLCVBackend):
    def __init__(self):
        try:
            from ib_insync import IB, util, Stock, Contract, Future
            self.IB = IB
            self.util = util
            self.Stock = Stock
            self.Contract = Contract
            self.Future = Future
        except Exception as e:
            raise RuntimeError("El backend 'ibkr' requiere instalar ib_insync") from e
        self.ib = self.IB()
        host = os.getenv("IB_HOST", "127.0.0.1")
        port = int(os.getenv("IB_PORT", "4002"))
        client_id = int(os.getenv("IB_CLIENT_ID", "7"))
        self.ib.connect(host, port, clientId=client_id)

    def history(self, ysymbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        raise NotImplementedError("Implementar mapping a contratos IBKR y llamada a reqHistoricalData")

def make_backend() -> OHLCVBackend:
    backend = os.getenv("DATA_BACKEND", "yahoo").lower()
    if backend == "yahoo":
        return YahooBackend()
    elif backend == "ibkr":
        return IBKRBackend()
    else:
        raise ValueError(f"DATA_BACKEND desconocido: {backend}")
