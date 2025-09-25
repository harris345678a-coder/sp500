# -*- coding: utf-8 -*-
"""
ranking.py - Módulo de Escaneo y Ranking de Mercado v2.2.0
---------------------------------------------------------------------
Este módulo es responsable de analizar un universo de activos financieros,
calcular indicadores técnicos clave y rankearlos para identificar las
mejores oportunidades de trading.

Diseño Robusto y Profesional:
- Obtención de datos a prueba de fallos con reintentos y backoff.
- Cálculo de indicadores de grado industrial, robusto contra NaNs y errores.
- Análisis tolerante a fallos para garantizar que el pipeline siempre se complete.
- Configuración flexible a través de variables de entorno.
- Lógica clara y modular para fácil mantenimiento.
"""
import os
import time
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ==============================================================================
# 1. CONFIGURACIÓN Y LOGGING
# ==============================================================================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
)
log = logging.getLogger("ranking_pro")

# --- Parámetros de la Estrategia (Configurables vía Env Vars) ---
MIN_ADX = float(os.getenv("MIN_ADX", "15.0"))
MIN_DOLLAR_VOL20 = float(os.getenv("MIN_DOLLAR_VOL20", "5000000.0"))  # $5M
FUTURES_LIQ_WHITELIST = {"GC=F", "CL=F"}

# ==============================================================================
# 2. LÓGICA DE DATOS (UNIVERSO Y DESCARGA)
# ==============================================================================
def get_universe() -> List[str]:
    """
    Define y devuelve el universo de tickers a analizar.
    Puede ser sobreescrito por la variable de entorno UNIVERSE_TICKERS.
    """
    custom_universe = os.getenv("UNIVERSE_TICKERS", "").strip()
    if custom_universe:
        tickers = [t.strip().upper() for t in custom_universe.replace(",", " ").split() if t]
        log.info(f"Using custom universe of {len(tickers)} tickers from ENV.")
        return tickers

    etfs_core = ["SPY","QQQ","IWM","DIA","VTI","VOO","IVV","VTV","VOE","VUG","VGT","VHT","VFH","VNQ","XLC","XLK","XLY","XLP","XLV","XLI","XLB","XLRE","XLU","XLF","XLE","SOXX","SMH","XME","GDX","GDXJ","IBB","XBI","IYR","IYT","XRT","XAR","XTL","GLD","SLV","DBC","DBA","USO","UNG","TLT","IEF","SHY","LQD","HYG","URA","TAN","OIH","XHB","ITB"]
    megacaps = ["AAPL","MSFT","NVDA","GOOGL","GOOG","AMZN","META","TSLA","AVGO","ADBE","CSCO","CRM","NFLX","AMD","INTC","QCOM","TXN","MU","AMAT","ASML","JPM","BAC","WFC","GS","MS","BLK","C","V","MA","AXP","INTU","XOM","CVX","COP","SLB","EOG","PSX","UNH","JNJ","LLY","ABBV","MRK","PFE","TMO","DHR","HD","LOW","COST","WMT","TGT","NKE","SBUX","MCD","BKNG","CAT","DE","BA","GE","HON","UPS","FDX","MMM","ORCL","SAP","PEP","KO","PG","CL","KHC","MDLZ","DIS","CMCSA","T","VZ"]
    growth_mid = ["NOW","PANW","SNOW","NET","ZS","DDOG","SHOP","SQ","PYPL","UBER","LYFT","WBD"]
    futures = ["GC=F", "CL=F"]
    
    seen = set()
    return [t for t in etfs_core + megacaps + growth_mid + futures if not (t in seen or seen.add(t))]

def download_with_retries(ticker: str, retries: int = 3, **kwargs) -> Optional[pd.DataFrame]:
    """
    Descarga datos de yfinance con reintentos y backoff exponencial.
    """
    for i in range(retries):
        try:
            df = yf.download(ticker, progress=False, **kwargs)
            if not df.empty:
                return df
            log.warning("No data found for ticker '%s'. It might be delisted.", ticker)
            return None
        except Exception as e:
            wait_time = 5 * (2**i)
            log.warning(
                "Download failed for '%s' (attempt %d/%d): %s. Retrying in %ds...",
                ticker, i + 1, retries, str(e).strip(), wait_time
            )
            time.sleep(wait_time)
            
    log.error("Could not download '%s' after %d attempts.", ticker, retries)
    return None

# ==============================================================================
# 3. CÁLCULO DE INDICADORES TÉCNICOS
# ==============================================================================
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _rma(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(alpha=1.0 / float(length), adjust=False).mean()

def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return _rma(tr, length)

def _adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Implementación de ADX de grado industrial, robusta y a prueba de fallos.
    Garantiza una salida de una sola columna (pd.Series).
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Movimiento Direccional
    move_up = high.diff()
    move_down = low.diff().mul(-1)
    
    plus_dm = pd.Series(
        np.where((move_up > move_down) & (move_up > 0), move_up, 0.0),
        index=df.index
    )
    minus_dm = pd.Series(
        np.where((move_down > move_up) & (move_down > 0), move_down, 0.0),
        index=df.index
    )

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    
    # Suavizado Wilder (RMA)
    atr = _rma(tr, length)
    plus_di = 100 * (_rma(plus_dm, length) / atr.replace(0, np.nan))
    minus_di = 100 * (_rma(minus_dm, length) / atr.replace(0, np.nan))
    
    # Índice Direccional (DX)
    di_diff_abs = (plus_di - minus_di).abs()
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (di_diff_abs / di_sum)
    
    # Índice Direccional Promedio (ADX)
    adx = _rma(dx.fillna(0), length)
    return adx

def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Añade todos los indicadores necesarios al DataFrame."""
    df_copy = df.copy()
    close = df_copy["Close"]
    df_copy['EMA20'] = _ema(close, 20)
    df_copy['EMA50'] = _ema(close, 50)
    df_copy['MOM63'] = close.pct_change(63)
    df_copy['ADX14'] = _adx(df_copy, 14)
    df_copy['DollarVol20'] = (close * df_copy['Volume']).rolling(20).mean()
    return df_copy

# ==============================================================================
# 4. LÓGICA DE ANÁLISIS Y RANKING
# ==============================================================================
def analyze_ticker(ticker: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analiza un único ticker y devuelve un diccionario con sus métricas.
    """
    try:
        if len(df) < 100:
            return {"ticker": ticker, "reason_excluded": "too_few_rows"}

        enriched_df = enrich_dataframe(df)
        latest = enriched_df.iloc[-1]
        
        dollar_vol = latest.get('DollarVol20', 0)
        dollar_vol = 0 if pd.isna(dollar_vol) else dollar_vol

        is_liquid = dollar_vol > MIN_DOLLAR_VOL20 or ticker in FUTURES_LIQ_WHITELIST
        if not is_liquid:
            return {"ticker": ticker, "reason_excluded": "illiquid"}

        if pd.isna(latest['EMA20']) or pd.isna(latest['EMA50']) or pd.isna(latest['MOM63']) or pd.isna(latest['ADX14']):
            return {"ticker": ticker, "reason_excluded": "indicator_nan"}

        long_trend = latest['EMA20'] > latest['EMA50'] and latest['MOM63'] > 0
        short_trend = latest['EMA20'] < latest['EMA50'] and latest['MOM63'] < 0

        if not long_trend and not short_trend:
             return {"ticker": ticker, "reason_excluded": "no_clear_trend"}

        is_strict = latest['ADX14'] > MIN_ADX
        side = "long" if long_trend else "short"
        
        score = latest['ADX14'] * abs(latest['MOM63'])

        return {
            "ticker": ticker,
            "side": side,
            "score": score if pd.notna(score) else 0.0,
            "is_strict": is_strict,
            "adx": latest['ADX14'],
            "last_close": latest['Close'],
            "atr14": _atr(df).iloc[-1]
        }
    except Exception as e:
        log.error(f"Analysis failed for {ticker}: {e}", exc_info=False)
        return {"ticker": ticker, "reason_excluded": "analysis_error"}

# ==============================================================================
# 5. PIPELINE PRINCIPAL
# ==============================================================================
def run_full_pipeline(**kwargs) -> Dict[str, Any]:
    """
    Orquesta todo el proceso: obtener universo, descargar datos, analizar y rankear.
    """
    t_start = time.time()
    universe = get_universe()
    log.info(f"Starting ranking pipeline for {len(universe)} tickers.")
    
    all_results = []
    fetched_count = 0
    
    for ticker in universe:
        df = download_with_retries(ticker, period="2y", interval="1d", auto_adjust=False)
        if df is None:
            all_results.append({"ticker": ticker, "reason_excluded": "download_failed"})
            continue
        
        fetched_count += 1
        analysis_result = analyze_ticker(ticker, df)
        if analysis_result:
            all_results.append(analysis_result)

    candidates = [r for r in all_results if "score" in r]
    excluded_count = len(all_results) - len(candidates)
    
    strict_candidates = sorted([c for c in candidates if c.get('is_strict')], key=lambda x: x.get('score', 0), reverse=True)
    relaxed_candidates = sorted([c for c in candidates if not c.get('is_strict')], key=lambda x: x.get('score', 0), reverse=True)

    top3_factors = strict_candidates[:3]
    
    top50_symbols = [c['ticker'] for c in strict_candidates]
    for c in relaxed_candidates:
        if len(top50_symbols) < 50 and c['ticker'] not in top50_symbols:
            top50_symbols.append(c['ticker'])
    
    t_elapsed = round(time.time() - t_start, 2)
    log.info(f"Pipeline finished in {t_elapsed}s. Found {len(strict_candidates)} strict candidates.")

    return {
        "ok": True,
        "took_s": t_elapsed,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "top50": top50_symbols[:50],
        "top3_factors": top3_factors,
        "diag": {
            "universe_count": len(universe),
            "fetched_count": fetched_count,
            "excluded_count": excluded_count,
            "strict_candidates_count": len(strict_candidates),
            "relaxed_candidates_count": len(relaxed_candidates),
        }
    }

# --- Bloque de Ejecución para Pruebas Locales ---
if __name__ == "__main__":
    log.info("Running local test of the ranking pipeline...")
    result = run_full_pipeline()
    print(json.dumps(result, indent=2))
