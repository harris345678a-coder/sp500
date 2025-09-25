# -*- coding: utf-8 -*-
"""
app_evolutivo.py - Orquestador de Pipeline Profesional v3.0.0
---------------------------------------------------------------------
Este módulo actúa como el orquestador principal del sistema. Es responsable de:
1. Exponer la API web.
2. Llamar al motor de ranking (`ranking.py`) para obtener candidatos.
3. Realizar una validación de calidad dinámica sobre los mejores candidatos.
4. Notificar a Telegram SOLO si existen señales de alta probabilidad.

Diseño Robusto y Profesional:
- Flujo de doble filtro: Escaneo amplio (ranking) + Validación estricta (aquí).
- Cero notificaciones de baja calidad: Se eliminó la lógica de fallback.
- Lógica tolerante a fallos para garantizar la finalización del pipeline.
"""
import os
import time
import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

# Importar el motor de ranking
try:
    import ranking
except ImportError:
    log.error("Error: ranking.py no se encuentra. Asegúrate de que esté en el mismo directorio.")
    # Permite que la app inicie, pero fallará en la ejecución.
    ranking = None

# --- Cliente de Redis (Opcional, para deduplicación) ---
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
log = logging.getLogger("app_evolutivo")

# --- Variables de Entorno ---
API_TOKEN = os.getenv("RUN_TOKEN", "123")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
REDIS_URL = os.getenv("REDIS_URL")

# ==============================================================================
# 2. LÓGICA DE ESTADO (DEDUPLICACIÓN DE NOTIFICACIONES)
# ==============================================================================
_redis_client = None

def get_redis_client() -> Optional['redis.Redis']:
    """Inicializa y devuelve un cliente de Redis si está configurado."""
    global _redis_client
    if not _REDIS_AVAILABLE or not REDIS_URL: return None
    if _redis_client is None:
        try:
            log.info("Connecting to Redis for deduplication...")
            _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            _redis_client.ping()
        except Exception as e:
            log.error("Failed to connect to Redis: %s", e)
            _redis_client = None
    return _redis_client

def read_last_digest() -> Optional[str]:
    """Lee el último digest de notificación desde Redis."""
    r = get_redis_client()
    if r:
        try: return r.get("evolutivo:last_digest")
        except Exception as e: log.error("Error reading from Redis: %s", e)
    return None

def write_last_digest(digest: str):
    """Escribe el último digest en Redis con una expiración de 7 días."""
    r = get_redis_client()
    if r:
        try: r.set("evolutivo:last_digest", digest, ex=60*60*24*7)
        except Exception as e: log.error("Error writing to Redis: %s", e)

# ==============================================================================
# 3. VALIDACIÓN DE SEÑALES DE ALTA CALIDAD
# ==============================================================================
def validate_and_build_signals(candidates: List[Dict]) -> List[Dict]:
    """
    Toma los mejores candidatos del ranking y los valida con reglas estrictas
    usando datos intradía para generar señales de alta calidad.
    """
    final_signals = []
    if not candidates:
        return final_signals

    log.info(f"Validating {len(candidates)} top candidates with intraday data...")
    for cand in candidates:
        symbol = cand.get("ticker")
        if not symbol: continue
        
        try:
            # Descargar datos de 60 minutos para un análisis más fino
            df_60m = ranking.download_with_retries(symbol, period="60d", interval="60m")
            if df_60m is None or df_60m.empty:
                log.warning(f"No 60m data for {symbol}, skipping validation.")
                continue

            price = df_60m['Close'].iloc[-1]
            atr = ranking._atr(df_60m, 14).iloc[-1]

            if not pd.notna(price) or not pd.notna(atr) or atr == 0:
                log.warning(f"Invalid price or ATR for {symbol}. Skipping.")
                continue

            # Lógica de riesgo/beneficio
            side = cand.get("side", "long")
            sl = price - 1.2 * atr if side == "long" else price + 1.2 * atr
            tp = price + 1.9 * atr if side == "long" else price - 1.9 * atr
            rr = abs(tp - price) / abs(price - sl) if abs(price - sl) > 1e-9 else 0

            # --- FILTRO DE CALIDAD FINAL ---
            # Solo se aprueba si el ratio riesgo/beneficio es aceptable.
            if rr > 1.5:
                final_signals.append({
                    "symbol": symbol,
                    "side": side,
                    "strategy": "Evolutivo Validated",
                    "trigger_price": round(price, 4),
                    "stop_loss": round(sl, 4),
                    "take_profit": round(tp, 4),
                    "risk_reward_ratio": round(rr, 2)
                })
            else:
                log.info(f"Candidate {symbol} rejected due to low RR: {rr:.2f}")

        except Exception as e:
            log.error(f"Validation failed for {symbol}: {e}", exc_info=False)

    return final_signals

# ==============================================================================
# 4. LÓGICA DE NOTIFICACIÓN A TELEGRAM
# ==============================================================================
def _escape_html(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _format_signal_msg(signal: Dict[str, Any]) -> str:
    """Formatea una única señal en un mensaje HTML para Telegram."""
    sym = _escape_html(signal.get("symbol", "?"))
    side = _escape_html(signal.get("side", "")).upper()
    strat = _escape_html(signal.get("strategy", "Validated"))
    
    header = f"<b>{sym} • {side} • {strat}</b>"
    
    levels = [
        f"TG: {signal.get('trigger_price', 'N/A')}",
        f"SL: {signal.get('stop_loss', 'N/A')}",
        f"TP: {signal.get('take_profit', 'N/A')}",
        f"RR: {signal.get('risk_reward_ratio', 'N/A')}"
    ]
    return f"{header}\n{' | '.join(levels)}"

def send_telegram_notification(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Construye y envía la notificación a Telegram."""
    if not BOT_TOKEN or not CHAT_ID:
        return {"sent": False, "reason": "telegram_not_configured"}
        
    if not signals:
        log.info("No high-quality signals found after validation. No notification will be sent.")
        return {"sent": False, "reason": "no_signals_approved"}

    messages = [_format_signal_msg(s) for s in signals]
    full_message_text = "\n\n".join(messages)
    
    # Deduplicación para evitar spam si se ejecuta varias veces
    new_digest = hashlib.sha256(full_message_text.encode("utf-8")).hexdigest()
    if read_last_digest() == new_digest:
        log.info("Skipping Telegram notification: duplicate content.")
        return {"sent": False, "reason": "duplicate_content"}

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": full_message_text, "parse_mode": "HTML"}
    
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        log.info(f"Successfully sent {len(signals)} signal(s) to Telegram.")
        write_last_digest(new_digest)
        return {"sent": True, "count": len(signals)}
    except requests.RequestException as e:
        log.error("Failed to send notification to Telegram: %s", e)
        return {"sent": False, "reason": f"http_error"}

# ==============================================================================
# 5. ORQUESTADOR PRINCIPAL Y API (FastAPI)
# ==============================================================================
app = FastAPI(title="App Evolutivo - Orquestador", version="3.0.0")

def run_complete_pipeline() -> Dict[str, Any]:
    """Función principal que orquesta la ejecución del pipeline completo."""
    t_start = time.time()
    
    # 1. Ejecutar el motor de escaneo y ranking
    if not ranking:
        raise HTTPException(status_code=500, detail="Ranking module not loaded.")
    ranking_payload = ranking.run_full_pipeline()
    if not ranking_payload.get("ok"):
        raise HTTPException(status_code=500, detail=f"Ranking pipeline failed: {ranking_payload.get('error')}")
    
    top3_candidates = ranking_payload.get("top3_factors", [])

    # 2. Validar los mejores candidatos para generar señales de alta calidad
    final_signals = validate_and_build_signals(top3_candidates)
    
    # 3. Enviar notificación SOLO si hay señales validadas
    tg_result = send_telegram_notification(final_signals)

    t_elapsed = round(time.time() - t_start, 2)
    log.info(f"Complete pipeline finished in {t_elapsed} seconds.")
    
    return {
        "ok": True,
        "elapsed_seconds": t_elapsed,
        "as_of": ranking_payload.get("as_of"),
        "final_signals": final_signals,
        "diag": {
            "ranking_summary": ranking_payload.get("diag", {}),
            "validation_summary": {
                "candidates_evaluated": len(top3_candidates),
                "signals_approved": len(final_signals),
            },
            "telegram_notification": tg_result,
        }
    }

@app.get("/healthz")
def healthz():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/rank/run-top3", summary="Run Full Pipeline and Notify")
def run_top3_endpoint(token: Optional[str] = Query(default=None)):
    """
    Ejecuta el pipeline de ranking, valida los 3 mejores y notifica
    únicamente si se encuentran señales de alta calidad.
    """
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API token")
    
    try:
        result = run_complete_pipeline()
        return JSONResponse(content=result)
    except Exception as e:
        log.critical("An unhandled exception occurred during pipeline execution: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")
