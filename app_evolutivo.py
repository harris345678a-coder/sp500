# -*- coding: utf-8 -*-
"""
App Evolutivo - Versión Profesional y Robusta
Diseñado para una ejecución fiable en plataformas como Render.

Características Clave:
- Obtención de datos con reintentos y backoff exponencial para máxima fiabilidad.
- Análisis de datos tolerante a fallos para procesar la mayor cantidad de tickers posible.
- Lógica de negocio defensiva para evitar caídas por datos malformados.
- Gestión de estado persistente (deduplicación) adaptable a Redis para entornos efímeros.
- Código limpio, organizado y con logging mejorado.
"""
import os
import json
import math
import time
import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

# --- Dependencias de Terceros ---
# Asegúrate de tener estas librerías en tu requirements.txt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse

# Intenta importar redis, pero no falles si no está.
try:
    import redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False

# ==============================================================================
# 1. CONFIGURACIÓN Y LOGGING
# ==============================================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
)
log = logging.getLogger("app_evolutivo_pro")

# --- Configuración de la App ---
API_TOKEN = os.getenv("RUN_TOKEN", "123") # CORREGIDO: Usando RUN_TOKEN como en tu imagen
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
REDIS_URL = os.getenv("REDIS_URL") # Render inyecta esta variable si añades el Add-on

# ==============================================================================
# 2. CLIENTE DE REDIS (Para estado persistente)
# ==============================================================================

_redis_client = None

def get_redis_client() -> Optional['redis.Redis']:
    """Inicializa y devuelve un cliente de Redis si está configurado."""
    global _redis_client
    if not _REDIS_AVAILABLE or not REDIS_URL:
        return None
    if _redis_client is None:
        try:
            log.info("Connecting to Redis...")
            _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            _redis_client.ping() # Verifica la conexión
        except Exception as e:
            log.error("Failed to connect to Redis: %s", e)
            _redis_client = None # Evita reintentos fallidos
    return _redis_client

def read_last_digest() -> Optional[str]:
    """Lee el último digest desde Redis."""
    r = get_redis_client()
    if not r:
        return None
    try:
        return r.get("evolutivo:last_digest")
    except Exception as e:
        log.error("Error reading from Redis: %s", e)
        return None

def write_last_digest(digest: str):
    """Escribe el último digest en Redis con una expiración de 7 días."""
    r = get_redis_client()
    if not r:
        return
    try:
        # TTL de 7 días (en segundos) para que no se acumule eternamente
        r.set("evolutivo:last_digest", digest, ex=60 * 60 * 24 * 7)
    except Exception as e:
        log.error("Error writing to Redis: %s", e)

# ==============================================================================
# 3. OBTENCIÓN DE DATOS ROBUSTA
# ==============================================================================

def download_with_retries(ticker: str, retries: int = 3, delay_secs: int = 5, **kwargs) -> Optional[pd.DataFrame]:
    """
    Descarga datos de yfinance con reintentos y backoff exponencial para
    manejar errores de red comunes en entornos de nube.
    """
    for i in range(retries):
        try:
            df = yf.download(ticker, progress=False, **kwargs)
            if df.empty:
                log.warning("No data found for ticker '%s'. It might be delisted.", ticker)
                return None
            return df
        except Exception as e:
            # Si falla, espera y vuelve a intentar.
            wait_time = delay_secs * (2**i)  # Backoff: 5s, 10s, 20s
            log.warning(
                "Failed to download '%s' (attempt %d/%d): %s. Retrying in %ds...",
                ticker, i + 1, retries, str(e).strip(), wait_time
            )
            time.sleep(wait_time)
    
    log.error("Could not download '%s' after %d attempts.", ticker, retries)
    return None

# ==============================================================================
# 4. LÓGICA DE ANÁLISIS Y CÁLCULO DE INDICADORES
# ==============================================================================

# (Las funciones de indicadores como _ema, _atr, _adx no necesitan cambios)
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _rma(series: pd.Series, length: int) -> pd.Series:
    alpha = 1.0 / float(length)
    return series.ewm(alpha=alpha, adjust=False).mean()

def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["High"].astype(float), df["Low"].astype(float), df["Close"].astype(float)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return _rma(tr, length)

def _adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high, low, close = df["High"].astype(float), df["Low"].astype(float), df["Close"].astype(float)
    prev_high, prev_low = high.shift(1), low.shift(1)
    plus_dm = (high - prev_high).where((high - prev_high) > (prev_low - low), 0.0).where((high-prev_high) > 0, 0.0)
    minus_dm = (prev_low - low).where((prev_low - low) > (high - prev_high), 0.0).where((prev_low-low) > 0, 0.0)
    
    tr_raw = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    tr_rma = _rma(tr_raw, length)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100.0 * (_rma(plus_dm, length) / tr_rma)
        minus_di = 100.0 * (_rma(minus_dm, length) / tr_rma)
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    
    return _rma(dx.fillna(0), length)


def _analyze_one_ticker(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Analiza un único ticker de forma segura, manejando errores de descarga y cálculo.
    """
    try:
        df = download_with_retries(ticker, period="2y", interval="1d")
        if df is None:
            return None # Falla en la descarga ya fue logueada

        close = df["Close"].astype(float)
        ema_fast = _ema(close, 20)
        ema_slow = _ema(close, 50)
        mom = close.pct_change(63) * 100.0  # ~3 meses
        adx = _adx(df, 14)
        atr = _atr(df, 14)
        
        # --- Extracción Segura de Valores ---
        # Usa .item() para convertir de forma segura una Serie de un elemento a un escalar
        last_price = close.iloc[-1].item()
        last_ema_fast = ema_fast.iloc[-1].item()
        last_ema_slow = ema_slow.iloc[-1].item()
        last_mom = mom.iloc[-1].item() if pd.notna(mom.iloc[-1]) else 0.0
        last_adx = adx.iloc[-1].item() if pd.notna(adx.iloc[-1]) else 0.0
        last_atr = abs(atr.iloc[-1].item()) if pd.notna(atr.iloc[-1]) else 0.0
        
        # --- Lógica de Decisión ---
        is_long_candidate = last_ema_fast > last_ema_slow and last_mom > 0 and last_adx >= 15.0
        side = "long" if is_long_candidate else "short"

        # --- Cálculo de Niveles ---
        sl = last_price - 1.2 * last_atr if side == "long" else last_price + 1.2 * last_atr
        tp1 = last_price + 1.0 * last_atr if side == "long" else last_price - 1.0 * last_atr
        tp2 = last_price + 1.9 * last_atr if side == "long" else last_price - 1.9 * last_atr
        
        risk = abs(last_price - sl)
        reward = abs(tp2 - last_price)
        rr = reward / risk if risk > 1e-9 else 0.0
        
        return {
            "symbol": ticker, "side": side, "strategy": "Evolutivo/Fallback",
            "tg": round(last_price, 4), "sl": round(sl, 4),
            "tp1": round(tp1, 4), "tp2": round(tp2, 4),
            "rr": round(rr, 2), "atr14": round(last_atr, 4),
        }

    except Exception as e:
        # Si CUALQUIER cosa falla para un ticker, lo logueamos y continuamos con los demás.
        log.error("Analysis failed for ticker '%s': %s", ticker, e, exc_info=True)
        return None

# ==============================================================================
# 5. LÓGICA DE NEGOCIO Y REGLAS DE PRE-ENVÍO
# ==============================================================================

def _normalize_signal(item: Any) -> Dict[str, Any]:
    """Asegura que cualquier objeto de señal tenga una estructura de diccionario consistente."""
    if isinstance(item, dict):
        d = dict(item)
        code = d.get("code") or d.get("symbol") or d.get("ticker") or d.get("ysymbol")
        if code:
            d.setdefault("symbol", code)
        return d
    if isinstance(item, str):
        return {"symbol": item}
    log.warning("Item in signal list has unexpected type: %s", type(item))
    return {"symbol": str(item)}

def _use_presend_rules(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ejecuta el módulo 'presend_rules' de forma segura y devuelve un resultado estructurado.
    """
    approved_signals, rejected_signals = [], []
    try:
        import presend_rules # Asume que este módulo existe en tu proyecto
        
        top3_factors = (payload or {}).get("top3_factors", [])
        
        # La función build_top3_signals ahora devuelve dos listas
        approved, rejected = presend_rules.build_top3_signals(top3_factors)
        
        approved_signals = [_normalize_signal(s) for s in approved]
        rejected_signals = rejected

    except ImportError:
        log.error("'presend_rules.py' not found. Cannot apply pre-send rules.")
    except Exception as e:
        log.error("An error occurred in presend_rules.build_top3_signals: %s", e, exc_info=True)
    
    return {
        "approved": approved_signals,
        "rejected": rejected_signals
    }

# ==============================================================================
# 6. NOTIFICACIONES A TELEGRAM
# ==============================================================================

def _escape_html(text: str) -> str:
    """Escapa caracteres HTML para enviar mensajes de forma segura."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _format_signal_msg(signal: Dict[str, Any]) -> str:
    """Formatea una única señal en un mensaje HTML para Telegram."""
    sym = _escape_html(signal.get("symbol", "?"))
    side = _escape_html(str(signal.get("side", "")).upper())
    strat = _escape_html(signal.get("strategy", "Evolutivo"))

    header = f"<b>{sym} • {side} • {strat}</b>"
    
    levels = []
    for key, label in [("tg", "TG"), ("sl", "SL"), ("tp1", "TP1"), ("tp2", "TP2"), ("rr", "RR"), ("atr14", "ATR")]:
        if key in signal:
            levels.append(f"{label}: {signal[key]}")
    
    return f"{header}\n{' | '.join(levels)}"

def send_telegram_notification(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Construye y envía la notificación a Telegram, manejando la deduplicación."""
    if not BOT_TOKEN or not CHAT_ID:
        return {"sent": False, "reason": "telegram_not_configured"}

    if not signals:
        return {"sent": False, "reason": "no_signals_to_send"}

    messages = [_format_signal_msg(s) for s in signals]
    full_message_text = "\n\n".join(messages)
    
    # --- Deduplicación vía Redis ---
    new_digest = hashlib.sha256(full_message_text.encode("utf-8")).hexdigest()
    last_digest = read_last_digest()

    if last_digest and last_digest == new_digest:
        log.info("Skipping Telegram notification: duplicate content.")
        return {"sent": False, "reason": "duplicate_content"}

    # --- Envío del Mensaje ---
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": full_message_text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status() # Lanza un error para respuestas 4xx/5xx
        log.info("Successfully sent notification to Telegram.")
        write_last_digest(new_digest)
        return {"sent": True, "count": len(signals)}
    except requests.RequestException as e:
        log.error("Failed to send notification to Telegram: %s", e)
        return {"sent": False, "reason": f"http_error: {e}"}

# ==============================================================================
# 7. ORQUESTADOR PRINCIPAL Y ENDPOINTS DE API
# ==============================================================================

app = FastAPI(title="App Evolutivo Pro", version="2.0.0")

def run_and_notify_pipeline() -> Dict[str, Any]:
    """
    Función principal que orquesta la ejecución del pipeline completo.
    """
    t_start = time.time()
    
    # --- 1. Ejecutar el pipeline de ranking principal ---
    try:
        import ranking # Asume que ranking.py existe
        raw_payload = ranking.run_full_pipeline()
    except Exception as e:
        log.critical("The main 'ranking.run_full_pipeline' failed: %s", e, exc_info=True)
        # No se puede continuar si el paso principal falla.
        raise HTTPException(status_code=500, detail=f"Ranking pipeline failed: {e}") from e

    # --- 2. Aplicar reglas de pre-envío para obtener señales finales ---
    presend_results = _use_presend_rules(raw_payload)
    final_signals = presend_results["approved"]
    used_presend_rules = True
    
    # --- 3. Fallback: si no hay señales, analiza el TOP 3 del ranking ---
    if not final_signals:
        log.warning("No signals approved by 'presend_rules'. Running fallback analysis.")
        used_presend_rules = False
        top3_items = (raw_payload or {}).get("top3_factors", [])[:3]
        top3_tickers = [str(_normalize_signal(item).get("symbol", "")) for item in top3_items]
        
        final_signals = [
            signal for ticker in top3_tickers if ticker
            if (signal := _analyze_one_ticker(ticker)) is not None
        ]

    # --- 4. Enviar notificación a Telegram ---
    tg_result = send_telegram_notification(final_signals)

    # --- 5. Construir y devolver la respuesta final ---
    t_elapsed = round(time.time() - t_start, 2)
    log.info("Pipeline finished in %s seconds.", t_elapsed)
    
    ranking_diag = raw_payload.get("diag", {})
    
    return {
        "ok": True,
        "elapsed_seconds": t_elapsed,
        "as_of": raw_payload.get("as_of", datetime.utcnow().isoformat()),
        "final_signals": final_signals,
        "diag": {
            "pipeline_flow": {
                "used_presend_rules": used_presend_rules,
                "used_fallback_analysis": not used_presend_rules,
            },
            "ranking_summary": {
                "universe_count": ranking_diag.get("universe_count"),
                "fetched_count": ranking_diag.get("fetched_count"),
                "strict_candidates": ranking_diag.get("strict_candidates_count"),
                "relaxed_candidates": ranking_diag.get("relaxed_candidates_count"),
                "top50_found": len(raw_payload.get("top50", [])),
                "top3_factors_found": len(raw_payload.get("top3_factors", [])),
            },
            "presend_rules_summary": {
                "approved_count": len(presend_results["approved"]),
                "rejected_count": len(presend_results["rejected"]),
                "rejection_samples": presend_results["rejected"][:3], # Muestra de rechazos
            },
            "telegram_notification": tg_result,
        }
    }

@app.get("/healthz")
def healthz():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/rank/run-top3", summary="Run Ranking Pipeline and Notify")
def run_top3_endpoint(token: Optional[str] = Header(default=None)):
    """
    Executes the full ranking pipeline, applies business rules,
    and sends a notification with the final signals.
    Authentication is required via 'token' header.
    """
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API token")
    
    try:
        result = run_and_notify_pipeline()
        return JSONResponse(content=result)
    except HTTPException as http_exc:
        # Re-lanza las excepciones HTTP que ya hemos manejado
        raise http_exc
    except Exception as e:
        # Captura cualquier otro error inesperado durante la ejecución
        log.critical("An unhandled exception occurred during pipeline execution: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

