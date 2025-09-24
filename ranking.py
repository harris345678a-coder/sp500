# ranking.py — Pipeline con logs de datos quirúrgicos (sin duplicados, profesional)
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger("ranking")

# ---------- Utilidades ----------
def _read_universe() -> List[str]:
    # 1) ENV: UNIVERSE_CSV="AAPL,MSFT,SPY"
    env_csv = os.getenv("UNIVERSE_CSV")
    if env_csv:
        items = [s.strip().upper() for s in env_csv.split(",") if s.strip()]
        if items:
            return items

    # 2) symbols.yaml (si existe, busca claves 'equities' o 'tickers')
    import yaml
    yaml_path = Path("symbols.yaml")
    if yaml_path.exists():
        try:
            data = yaml.safe_load(yaml_path.read_text())
            for key in ("equities", "tickers", "symbols"):
                if key in data and isinstance(data[key], list):
                    return [str(s).upper() for s in data[key]]
        except Exception:
            log.exception("No pude leer symbols.yaml, usando universo por defecto")

    # 3) Fallback mínimo pero útil
    return ["SPY", "QQQ", "IWM", "AAPL", "NVDA", "AMZN", "GLD", "SLV", "SOXX", "SMH", "GDX", "DBC", "DBA", "XLK", "SHY"]

def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) + 1e-12)

def _audit_line(tag: str, **kv) -> None:
    # Línea compacta para buscar rápidamente en logs
    # ej: 2025-09-24 03:16:54 | INFO | ranking | DATA_AUDIT | {...}
    log.info("DATA_AUDIT | %s", {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in kv.items()})

# ---------- Descarga ----------
def _download_hist(ticker: str, period="2y", interval="1d") -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, actions=True, prepost=False)
        # Normaliza columnas
        if not df.empty:
            df = df.reset_index().rename(columns={"index": "Date"})
            if "Datetime" in df.columns:  # yfinance usa "Date" o "Datetime" según intervalo
                df.rename(columns={"Datetime": "Date"}, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_convert("America/New_York").dt.tz_localize(None, ambiguous="NaT", nonexistent="NaT") if hasattr(df["Date"].dt, "tz") else pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
    except Exception:
        log.exception("Error descargando datos", extra={"ticker": ticker})
        return pd.DataFrame()
    return df

# ---------- Features / Ranking ----------
def _build_score(df: pd.DataFrame) -> pd.Series:
    # Señales simples: momentum 63d y volatilidad 21d (baja vol = mejor)
    close = df["Close"].astype(float)
    ret_63 = close.pct_change(63)
    vol_21 = close.pct_change().rolling(21).std()
    score = _zscore(ret_63) - _zscore(vol_21)
    return score

def compute_top50(universe: List[str], audit: bool=False) -> List[str]:
    rows = []
    for t in universe:
        d = _download_hist(t, period="2y", interval="1d")
        if audit:
            n = int(d.shape[0])
            start = d.index.min().date().isoformat() if n else None
            end = d.index.max().date().isoformat() if n else None
            nan_close = int(d["Close"].isna().sum()) if "Close" in d.columns else n
            _audit_line("daily", ticker=t, rows=n, start=start, end=end, nan_close=nan_close)
        if d.empty or "Close" not in d:
            continue
        rows.append((t, _build_score(d).iloc[-1]))

    if not rows:
        log.warning("compute_top50: sin datos utilizables")
        return []

    # Orden descendente por score y corta a 50
    rows.sort(key=lambda x: (x[1] if pd.notna(x[1]) else -np.inf), reverse=True)
    top = [t for t, s in rows if pd.notna(s)]
    top = top[:50]
    if audit:
        _audit_line("top50", count=len(top))
    return top

@dataclass
class FactorSignal:
    ticker: str
    reasons: List[str]

def compute_top3_factors(top50: List[str], audit: bool=False) -> List[FactorSignal]:
    # Señales muy simples a modo de ejemplo auditable: ruptura de 55d + EMA(20) > EMA(50)
    def ema(s: pd.Series, span: int) -> pd.Series:
        return s.ewm(span=span, adjust=False).mean()

    scored = []
    for t in top50:
        d = _download_hist(t, period="1y", interval="1d")
        if d.empty or "Close" not in d:
            continue
        c = d["Close"].astype(float)
        hh = c.rolling(55).max()
        e20, e50 = ema(c, 20), ema(c, 50)
        cond_breakout = c.iloc[-1] >= hh.iloc[-2] if pd.notna(hh.iloc[-2]) else False
        cond_trend = e20.iloc[-1] > e50.iloc[-1] if pd.notna(e20.iloc[-1]) and pd.notna(e50.iloc[-1]) else False
        score = 0
        reasons = []
        if cond_breakout:
            score += 1; reasons.append("breakout55")
        if cond_trend:
            score += 1; reasons.append("ema20>ema50")
        if score > 0:
            scored.append((t, score, reasons))

    # Ordena por score y se queda con los 3 primeros
    scored.sort(key=lambda x: (x[1], x[0]), reverse=True)
    result = [FactorSignal(ticker=t, reasons=r) for t, s, r in scored[:3]]
    if audit:
        _audit_line("top3_summary", count=len(result), tickers=[x.ticker for x in result])
    return result

# ---------- Orquestador ----------
def run_full_pipeline(audit: bool | None = None) -> Dict:
    audit = bool(audit) or os.getenv("AUDIT", "").lower() in {"1", "true", "on", "yes", "y"}
    uni = _read_universe()

    if audit:
        log.info("DATA_AUDIT | universe | %s", {"count": len(uni), "tickers_sample": uni[:10]})

    top50 = compute_top50(uni, audit=audit)
    top3_objs = compute_top3_factors(top50, audit=audit)

    payload = {
        "as_of": datetime.now(timezone.utc).date().isoformat(),
        "top50": top50,
        "top3_factors": [{"ticker": x.ticker, "reasons": x.reasons} for x in top3_objs],
    }

    # Resumen final para búsquedas rápidas
    log.info("RESUMEN | as_of=%s | top50=%s | top3=%s", payload["as_of"], len(top50), len(payload["top3_factors"]))
    return payload