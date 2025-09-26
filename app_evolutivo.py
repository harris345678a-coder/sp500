# -*- coding: utf-8 -*-
"""
app_evolutivo.py - Orquestador con Validación Profesional v4.3.0
---------------------------------------------------------------------
Este módulo implementa una lógica de validación de señales de nivel
profesional sobre los 50 MEJORES candidatos del ranking.

NUEVA LÓGICA DE TRABAJO:
1.  Obtiene los 50 mejores candidatos del motor de `ranking.py`.
2.  Itera sobre estos 50 y somete a cada uno al "Checklist de Despegue".
3.  De todos los que PASAN la validación, los ordena por el mejor
    ratio riesgo/beneficio y selecciona los 3 MEJORES.
4.  Estos "Tres Finales" son los que se guardan y notifican.
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
    # Permite que la app inicie, pero fallará en la ejecución.
    ranking = None

# --- Cliente de Redis ---
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

API_TOKEN = os.getenv("RUN_TOKEN", "123")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
REDIS_URL = os.getenv("REDIS_URL")
ACTIVE_SIGNALS_KEY = "evolutivo:active_signals"

# ==============================================================================
# 2. LÓGICA DE ESTADO (Redis)
# ==============================================================================
_redis_client = None
def get_redis_client() -> Optional['redis.Redis']:
    """Inicializa y devuelve un cliente de Redis si está configurado."""
    global _redis_client
    if not _REDIS_AVAILABLE or not REDIS_URL: return None
    if _redis_client is None:
        try:
            _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            _redis_client.ping()
        except Exception as e:
            log.error(f"Failed to connect to Redis: {e}")
            _redis_client = None
    return _redis_client

def update_signal_in_db(signal: Dict):
    """Guarda o actualiza una señal en la base de datos de Redis."""
    r = get_redis_client()
    if not r: return
    signal_id = f"{signal['symbol']}:{signal['side']}"
    try:
        r.hset(ACTIVE_SIGNALS_KEY, signal_id, json.dumps(signal))
    except Exception as e:
        log.error(f"Failed to update signal {signal_id} in Redis: {e}")

# ==============================================================================
# 3. LÓGICA DE VALIDACIÓN DE SEÑALES PROFESIONAL
# ==============================================================================
def validate_and_build_signals(candidates: List[Dict]) -> List[Dict]:
    """
    Toma una lista de candidatos y los somete a un checklist de validación
    de breakout de nivel profesional usando datos de 60 minutos.
    """
    validated_signals = []
    if not candidates:
        return validated_signals

    log.info(f"Starting professional validation for up to {len(candidates)} candidates...")
    for i, cand in enumerate(candidates):
        symbol = cand.get("ticker")
        side = cand.get("side")
        if not symbol or not side:
            continue
        
        log.info(f"({i+1}/{len(candidates)}) Validating {symbol} ({side})...")
        try:
            # 1. OBTENER DATOS INTRADÍA
            df_60m = ranking.download_with_retries(symbol, period="60d", interval="60m")
            if df_60m is None or len(df_60m) < 22: # Minimo para EMA21 + velas de pivot
                log.warning(f"[{symbol}] Insufficient 60m data for validation. Skipping.")
                continue

            # 2. CALCULAR INDICADORES INTRADÍA
            df_60m['EMA8'] = ranking._ema(df_60m['Close'], 8)
            df_60m['EMA21'] = ranking._ema(df_60m['Close'], 21)
            df_60m['ATR14'] = ranking._atr(df_60m, 14)
            df_60m['AvgVol20'] = df_60m['Volume'].rolling(20).mean()
            
            latest = df_60m.iloc[-1]
            price, atr = latest['Close'], latest['ATR14']
            
            required_values = [price, atr, latest['EMA8'], latest['EMA21'], latest['AvgVol20']]
            if any(pd.isna(v) for v in required_values):
                log.warning(f"[{symbol}] Indicators have NaN values. Skipping.")
                continue

            # 3. CHECKLIST DE VALIDACIÓN DE BREAKOUT
            if side == 'long':
                pivot_level = df_60m['High'].iloc[-6:-1].max()
                price_ok = price > pivot_level
                volume_ok = latest['Volume'] > latest['AvgVol20']
                trend_ok = latest['EMA8'] > latest['EMA21']
                extension_ok = (price - latest['EMA21']) < (1.5 * atr)
            else: # side == 'short'
                pivot_level = df_60m['Low'].iloc[-6:-1].min()
                price_ok = price < pivot_level
                volume_ok = latest['Volume'] > latest['AvgVol20']
                trend_ok = latest['EMA8'] < latest['EMA21']
                extension_ok = (latest['EMA21'] - price) < (1.5 * atr)

            log.info(f"[{symbol}] Checklist | Price OK: {price_ok}, Volume OK: {volume_ok}, Trend OK: {trend_ok}, Extension OK: {extension_ok}")

            # 4. APROBAR SEÑAL SI TODO ES CORRECTO
            if price_ok and volume_ok and trend_ok and extension_ok:
                sl = price - 1.2 * atr if side == "long" else price + 1.2 * atr
                tp1 = price + 1.0 * atr if side == "long" else price - 1.0 * atr
                tp2 = price + 1.9 * atr if side == "long" else price - 1.9 * atr
                rr = abs(tp2 - price) / abs(price - sl) if abs(price - sl) > 1e-9 else 0

                validated_signals.append({
                    "symbol": symbol, "side": side, "strategy": "Breakout Validated",
                    "trigger_price": round(price, 4), "stop_loss": round(sl, 4),
                    "take_profit_1": round(tp1, 4), "take_profit_2": round(tp2, 4),
                    "risk_reward_ratio": round(rr, 2)
                })
                log.info(f"[{symbol}] PASSED validation. Added to potential signals.")
        except Exception as e:
            log.error(f"[{symbol}] An unexpected error occurred during validation: {e}", exc_info=False)
            
    return validated_signals

# ==============================================================================
# 4. ORQUESTADOR Y API
# ==============================================================================
app = FastAPI(title="App Evolutivo - Sistema Profesional", version="4.3.0")

def send_new_signal_notification(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Notifica sobre la creación de NUEVAS señales de alta calidad."""
    if not BOT_TOKEN or not CHAT_ID: return {"sent": False, "reason": "telegram_not_configured"}
    if not signals:
        log.info("No high-quality signals found after validation. No notification will be sent.")
        return {"sent": False, "reason": "no_signals_approved"}

    messages = [f"<b>🚨 NUEVA SEÑAL: {s['symbol']} ({s['side'].upper()})</b>\nEntrada: {s['trigger_price']} | SL: {s['stop_loss']} | TP1: {s['take_profit_1']}" for s in signals]
    full_message_text = "\n\n".join(messages)
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": full_message_text, "parse_mode": "HTML"}
    
    try:
        requests.post(url, json=payload, timeout=10).raise_for_status()
        log.info(f"Successfully sent {len(signals)} NEW signal(s) to Telegram.")
        return {"sent": True, "count": len(signals)}
    except Exception as e:
        log.error("Failed to send new signal notification: %s", e)
        return {"sent": False, "reason": "http_error"}

def run_discovery_pipeline() -> Dict:
    """
    Ejecuta el pipeline de descubrimiento, validando el Top 50 y seleccionando los 3 mejores.
    """
    log.info("--- Starting Discovery Phase ---")
    if not ranking:
        raise HTTPException(status_code=500, detail="Ranking module not loaded.")
    
    ranking_payload = ranking.run_full_pipeline()
    if not ranking_payload.get("ok"):
        raise HTTPException(status_code=500, detail="Ranking pipeline failed")
    
    # 1. Obtener los 50 mejores candidatos del motor de ranking
    top50_candidates = ranking_payload.get("top50_candidates", [])
    
    # 2. Validar la lista completa de candidatos
    validated_signals = validate_and_build_signals(top50_candidates)
    
    # 3. Seleccionar los 3 mejores de los validados, ordenados por RR
    final_top_3 = sorted(validated_signals, key=lambda x: x['risk_reward_ratio'], reverse=True)[:3]
    
    if final_top_3:
        for signal in final_top_3:
            signal['status'] = 'active'
            signal['created_at'] = datetime.now(timezone.utc).isoformat()
            update_signal_in_db(signal)
            
    tg_result = send_new_signal_notification(final_top_3)
    
    return {
        "ranking_diag": ranking_payload.get("diag", {}),
        "candidates_evaluated": len(top50_candidates),
        "signals_validated": len(validated_signals),
        "final_signals_selected": len(final_top_3),
        "telegram_notification": tg_result,
        "final_signals": final_top_3
    }

@app.get("/healthz")
def healthz():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/rank/run", summary="Run Full Discovery and Validation on Top 50")
def run_full_pipeline_endpoint(token: Optional[str] = Query(default=None)):
    if token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid API token")
    
    t_start = time.time()
    try:
        # Nota: En un sistema completo, la fase de monitoreo iría aquí primero.
        # run_monitoring_phase()
        discovery_results = run_discovery_phase()
        t_elapsed = round(time.time() - t_start, 2)
        log.info(f"Full pipeline finished in {t_elapsed}s.")

        return JSONResponse(content={
            "ok": True,
            "elapsed_seconds": t_elapsed,
            "as_of": datetime.now(timezone.utc).isoformat(),
            "discovery_summary": discovery_results,
        })
    except Exception as e:
        log.critical("An unhandled exception occurred during full pipeline execution: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")