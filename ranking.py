# -*- coding: utf-8 -*-
"""
ranking.py — pipeline robusto para ranking + top3.

Características clave
- Universo por defecto amplio y líquido (sin depender de CSV/URL).
- Opcional: universo por env var (lista), CSV local o CSV remoto.
- Descarga por lotes con yfinance.download() + fallback por símbolo con reintentos.
- Filtros estrictos: datos suficientes, frescura, EMA20>EMA50, momentum63>0,
  precio mínimo, liquidez por dólar-vol 20d. Sin bypass en fallback.
- Logs de auditoría: "DATA_AUDIT | ..." por símbolo y "RESUMEN | ...".
- API: run_full_pipeline(audit=False) -> dict compatible con app_evolutivo.

Requisitos: yfinance, pandas, numpy, requests.
"""
from __future__ import annotations

import os
import time
import json
import math
import logging
import datetime as dt
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

# Dependencia principal de datos
try:
    import yfinance as yf
except Exception as e:
    raise RuntimeError("Necesitas instalar yfinance") from e

try:
    import requests  # sólo si se usa CSV remoto
except Exception:
    requests = None

# ------------------------- Logging -------------------------

LOG_LEVEL = os.getenv("RANKING_LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("ranking")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# ------------------------- Universo por defecto -------------------------
# Universo amplio "embedded" (líquido y generalista).
# Incluye: índices/sectores (ETFs), commodities/bonos (ETFs), mega-caps,
# semis, energía, salud, consumo, bancos; + futuros GC=F y CL=F.
# (Sin cripto).
DEFAULT_UNIVERSE: List[str] = [
    # Index/market ETFs
    "SPY","QQQ","IWM","DIA","VTI","VOO","IVV","VTV","VOE","VUG","VGT","VHT","VFH","VNQ",
    # Sector SPDRs
    "XLK","XLE","XLF","XLY","XLP","XLV","XLI","XLB","XLRE","XLU","XLC",
    # Thematic / industry ETFs
    "SOXX","SMH","XME","GDX","GDXJ","IBB","XBI","IYR","IYT","KRE","XOP","OIH","XHB","ITB",
    "KWEB","HACK","CIBR","TAN","URA","XRT","XAR","XTL",
    # Commodities/Bonds ETFs
    "GLD","SLV","DBC","DBA","USO","UNG","TLT","IEF","SHY","LQD","HYG",
    # Megacaps / líderes liquidez
    "AAPL","MSFT","NVDA","GOOGL","GOOG","AMZN","META","TSLA","AVGO","ADBE","CSCO","CRM","NFLX",
    "AMD","INTC","QCOM","TXN","MU","AMAT","ASML",
    "JPM","BAC","WFC","GS","MS","BLK","C",
    "XOM","CVX","COP","SLB","EOG","PSX",
    "UNH","JNJ","LLY","ABBV","MRK","PFE","TMO","DHR",
    "HD","LOW","COST","WMT","TGT","NKE","SBUX","MCD","BKNG",
    "CAT","DE","BA","GE","HON","UPS","FDX","MMM",
    "KO","PEP","PG","CL","KHC","MDLZ",
    "ORCL","SAP","NOW","PANW","SNOW","NET","ZS","DDOG",
    "SHOP","SQ","PYPL","UBER","LYFT",
    "DIS","CMCSA","T","VZ","WBD",
    "V","MA","AXP","INTU",
    # Futuros (solo oro y petróleo, como solicitaste)
    "GC=F","CL=F",
]

def _dedupe_keep_order(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        if s not in seen and s is not None and s != "":
            seen.add(s); out.append(s)
    return out

def _load_universe_from_env() -> Optional[List[str]]:
    raw = os.getenv("RANKING_UNIVERSE")
    if raw:
        parts = [x.strip() for x in raw.split(",")]
        parts = [p for p in parts if p and not p.startswith("^") and "-USD" not in p]  # sin índices/cripto
        return _dedupe_keep_order(parts) or None
    return None

def _load_universe_from_csv_path() -> Optional[List[str]]:
    path = os.getenv("RANKING_UNIVERSE_CSV_PATH")
    if not path:
        return None
    try:
        df = pd.read_csv(path)
        col = "symbol" if "symbol" in df.columns else ("ticker" if "ticker" in df.columns else None)
        if not col:
            return None
        vals = [str(x).strip().upper() for x in df[col].tolist()]
        vals = [v for v in vals if v and not v.startswith("^") and "-USD" not in v]
        return _dedupe_keep_order(vals) or None
    except Exception as e:
        logger.warning(f"DATA_AUDIT | universo_csv_path_error | {e}")
        return None

def _load_universe_from_csv_url() -> Optional[List[str]]:
    url = os.getenv("RANKING_UNIVERSE_CSV_URL")
    if not url or requests is None:
        return None
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        col = "symbol" if "symbol" in df.columns else ("ticker" if "ticker" in df.columns else None)
        if not col:
            return None
        vals = [str(x).strip().upper() for x in df[col].tolist()]
        vals = [v for v in vals if v and not v.startswith("^") and "-USD" not in v]
        return _dedupe_keep_order(vals) or None
    except Exception as e:
        logger.warning(f"DATA_AUDIT | universo_csv_url_error | {e}")
        return None

def build_universe() -> List[str]:
    """
    Orden de precedencia:
    1) RANKING_UNIVERSE (lista inline)
    2) RANKING_UNIVERSE_CSV_PATH (CSV local)
    3) RANKING_UNIVERSE_CSV_URL (CSV remoto)
    4) DEFAULT_UNIVERSE (embebido)
    Además: respeta tu restricción de futuros: incluye GC=F y CL=F; sin cripto.
    """
    for src in (_load_universe_from_env, _load_universe_from_csv_path, _load_universe_from_csv_url):
        vals = src()
        if vals:
            uni = vals
            break
    else:
        uni = DEFAULT_UNIVERSE[:]
    # saneo adicional
    uni = [u.strip().upper() for u in uni if u and "-USD" not in u]  # sin cripto
    uni = _dedupe_keep_order(uni)
    # aseguro futuros requeridos
    for fut in ("GC=F","CL=F"):
        if fut not in uni:
            uni.append(fut)
    # elimino índices (^)
    uni = [u for u in uni if not u.startswith("^")]
    sample = [x for x in uni[:10]]
    logger.info(f'DATA_AUDIT | universe | {json.dumps({"count": len(uni), "tickers_sample": sample})}')
    return uni

# ------------------------- Utilidades técnicas -------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def momentum_roc(series: pd.Series, window: int) -> pd.Series:
    return series.pct_change(window)

def wilder_smoothing(series: pd.Series, period: int) -> pd.Series:
    # Wilder EMA para ADX
    alpha = 1.0 / period
    return series.ewm(alpha=alpha, adjust=False).mean()

def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    # Implementación liviana de ADX (no requiere librerías externas)
    up_move = high.diff()
    down_move = low.diff().mul(-1.0)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = (high - low)
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = wilder_smoothing(tr, period)

    plus_di = 100 * wilder_smoothing(pd.Series(plus_dm, index=high.index), period) / atr
    minus_di = 100 * wilder_smoothing(pd.Series(minus_dm, index=high.index), period) / atr
    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) ) * 100
    adx = wilder_smoothing(dx, period)
    return adx

# ------------------------- Descarga robusta -------------------------

def download_batch(tickers: List[str], period: str = "2y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """
    Descarga por lotes. Devuelve dict por ticker con columnas ['Open','High','Low','Close','Adj Close','Volume'].
    Si yfinance devuelve panel multiindex, lo separamos por símbolo.
    """
    if not tickers:
        return {}

    # yfinance limita longitudes muy grandes; hacemos rebanadas
    out: Dict[str, pd.DataFrame] = {}
    step = int(os.getenv("YF_BATCH_SIZE", "60"))
    for i in range(0, len(tickers), step):
        chunk = tickers[i:i+step]
        try:
            df = yf.download(
                tickers=" ".join(chunk),
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                for sym in chunk:
                    if sym in df.columns.get_level_values(0):
                        sub = df[sym].copy()
                        sub.columns = [c.title().replace(" ", "") for c in sub.columns]  # homogeneiza
                        out[sym] = sub.dropna(how="all")
            else:
                # Caso de un solo símbolo
                sub = df.copy()
                sub.columns = [c.title().replace(" ", "") for c in sub.columns]
                out[chunk[0]] = sub.dropna(how="all")
        except Exception as e:
            logger.warning(f"DATA_AUDIT | batch_download_error | chunk={chunk[0]}.. | {e}")

    return out

def download_single(symbol: str, retries: int = 2, period: str = "2y", interval: str = "1d") -> Optional[pd.DataFrame]:
    backoff = 1.0
    for n in range(retries + 1):
        try:
            df = yf.download(
                tickers=symbol,
                period=period,
                interval=interval,
                group_by="ticker",
                auto_adjust=False,
                threads=False,
                progress=False,
            )
            if df is None or df.empty:
                raise RuntimeError("empty")
            df.columns = [c.title().replace(" ", "") for c in df.columns]
            return df.dropna(how="all")
        except Exception as e:
            if n < retries:
                time.sleep(backoff)
                backoff *= 2
            else:
                logger.warning(f"DATA_AUDIT | single_download_error | {symbol} | {e}")
                return None

# ------------------------- Evaluación y filtros -------------------------

def evaluate_symbol(sym: str, df: pd.DataFrame, today: dt.date, cfg: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Devuelve (result, audit_row)
    - result: dict con métricas y razones si pasa filtros; None si se excluye
    - audit_row: siempre presente para logging
    """
    audit = {
        "ticker": sym,
        "rows": int(df.shape[0]),
        "start": str(df.index.min().date()) if not df.empty else None,
        "end": str(df.index.max().date()) if not df.empty else None,
        "nan_close": int(df["Close"].isna().sum()) if "Close" in df else None,
        "reason_excluded": None,
    }

    # Validaciones de columnas
    need_cols = {"Open","High","Low","Close","Volume"}
    if not need_cols.issubset(set(df.columns)):
        audit["reason_excluded"] = "missing_cols"
        return None, audit

    # Ventana mínima de datos
    if df.shape[0] < cfg["min_rows"]:
        audit["reason_excluded"] = "insufficient_rows"
        return None, audit

    # Frescura (último día <= max_age_days)
    last_day = df.index.max().date()
    if (today - last_day).days > cfg["max_age_days"]:
        audit["reason_excluded"] = "stale"
        return None, audit

    # Cálculos técnicos
    close = df["Close"]
    high, low = df["High"], df["Low"]
    vol = df["Volume"].astype(float)

    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    mom63 = close.pct_change(63)  # ~3 meses
    adx14 = compute_adx(high, low, close, period=14)

    # Liquidez por dólar (20d)
    dollar_vol20 = (close * vol).rolling(20, min_periods=20).mean()

    # Precio mínimo
    if close.iloc[-1] < cfg["min_price"]:
        audit["reason_excluded"] = "price_too_low"
        return None, audit

    # Liquidez mínima
    if dollar_vol20.iloc[-1] < cfg["min_dollar_vol_20d"]:
        audit["reason_excluded"] = "illiquid"
        return None, audit

    # Filtros técnicos estrictos (sin bypass)
    if pd.isna(ema20.iloc[-1]) or pd.isna(ema50.iloc[-1]):
        audit["reason_excluded"] = "ema_nan"
        return None, audit
    if ema20.iloc[-1] <= ema50.iloc[-1]:
        audit["reason_excluded"] = "ema20<=ema50"
        return None, audit

    if pd.isna(mom63.iloc[-1]) or mom63.iloc[-1] <= 0:
        audit["reason_excluded"] = "momentum63<=0"
        return None, audit

    if pd.isna(adx14.iloc[-1]) or adx14.iloc[-1] < cfg["min_adx"]:
        audit["reason_excluded"] = "adx_low"
        return None, audit

    # Score para ranking (simple y estable)
    ema_ratio = float(ema20.iloc[-1] / ema50.iloc[-1])
    score = (mom63.iloc[-1] * 100.0) + (ema_ratio - 1.0) * 50.0 + (max(min((adx14.iloc[-1]-cfg["min_adx"]), 20), 0) * 0.5)

    result = {
        "ticker": sym,
        "score": float(score),
        "reasons": [
            "ema20>ema50",
            "momentum63_pos",
            f"adx>={cfg['min_adx']}",
            "avg_dollar_vol20_ok"
        ],
        "last_close": float(close.iloc[-1]),
        "ema20": float(ema20.iloc[-1]),
        "ema50": float(ema50.iloc[-1]),
        "mom63": float(mom63.iloc[-1]),
        "adx14": float(adx14.iloc[-1]),
        "dollar_vol20": float(dollar_vol20.iloc[-1]),
    }
    return result, audit

# ------------------------- Descarga -------------------------

def run_full_pipeline(audit: bool=False, **kwargs) -> Dict[str, Any]:
    """
    Ejecuta todo y devuelve payload para API.
    Parámetros (env):
      - RANKING_MIN_ROWS (por defecto 150)
      - RANKING_MAX_AGE_DAYS (por defecto 5)
      - RANKING_MIN_PRICE (por defecto 3.0)
      - RANKING_MIN_DOLLAR_VOL_20D (por defecto 1e7)
      - RANKING_MIN_ADX (por defecto 15.0)
      - RANKING_TOPN (por defecto 50)
    """
    t0 = time.time()
    today = dt.date.today()

    cfg = {
        "min_rows": int(os.getenv("RANKING_MIN_ROWS", "150")),
        "max_age_days": int(os.getenv("RANKING_MAX_AGE_DAYS", "5")),
        "min_price": float(os.getenv("RANKING_MIN_PRICE", "3.0")),
        "min_dollar_vol_20d": float(os.getenv("RANKING_MIN_DOLLAR_VOL_20D", "10000000")),  # 10M
        "min_adx": float(os.getenv("RANKING_MIN_ADX", "15.0")),
        "topn": int(os.getenv("RANKING_TOPN", "50")),
    }

    universe = build_universe()

    # Descarga por lotes
    data_map = download_batch(universe, period="2y", interval="1d")

    # Fallback por símbolo que no apareció en el lote
    missing = [s for s in universe if s not in data_map]
    for sym in missing:
        df = download_single(sym, retries=2, period="2y", interval="1d")
        if df is not None and not df.empty:
            data_map[sym] = df

    survivors: List[Dict[str, Any]] = []
    excluded_sample: List[Tuple[str,str]] = []

    # Auditoría por símbolo
    for sym in universe:
        df = data_map.get(sym)
        if df is None or df.empty:
            logger.info(f'DATA_AUDIT | {json.dumps({"ticker": sym, "rows": 0, "start": None, "end": None, "nan_close": None, "reason_excluded": "no_data"})}')
            if len(excluded_sample) < 5:
                excluded_sample.append((sym, "no_data"))
            continue
        res, audit_row = evaluate_symbol(sym, df, today, cfg)
        logger.info(f'DATA_AUDIT | {json.dumps(audit_row)}')
        if res is not None:
            survivors.append(res)
        else:
            if len(excluded_sample) < 5:
                excluded_sample.append((sym, str(audit_row.get("reason_excluded"))))

    # Ranking y selección
    survivors.sort(key=lambda x: (-x["score"], x["ticker"]))
    topn = survivors[: cfg["topn"]]
    top50_tickers = [x["ticker"] for x in topn]

    # Top3 (sin bypass de filtros)
    top3 = topn[:3]
    top3_factors = [{"ticker": x["ticker"], "reasons": x["reasons"][:]} for x in top3]

    # Resumen / diagnóstico
    fetched_count = len(data_map)
    excluded_count = len(universe) - len(survivors)
    diag = {
        "universe_count": len(universe),
        "fetched_count": fetched_count,
        "excluded_count": excluded_count,
        "excluded_sample": excluded_sample,
    }

    if audit or os.getenv("RANKING_DATA_AUDIT","false").lower() == "true":
        logger.info(f'DATA_AUDIT | {json.dumps({"count": len(survivors)})}')
        logger.info(f'DATA_AUDIT | {json.dumps({"count": len(top3), "tickers": [x["ticker"] for x in top3]})}')

    took = time.time() - t0
    logger.info(f'RESUMEN | as_of={today.isoformat()} | top50={len(topn)} | top3={len(top3)}')

    payload = {
        "ok": True,
        "took_s": round(took, 2),
        "as_of": today.isoformat(),
        "top50": top50_tickers,
        "top3_factors": top3_factors,
        "diag": diag,
    }
    return payload

# Modo CLI rápido: python -m ranking
if __name__ == "__main__":
    print(json.dumps(run_full_pipeline(audit=True), indent=2))
