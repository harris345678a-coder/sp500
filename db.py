import os, json, time
from typing import Optional

# Stubs seguros para entorno Render Free (sin Postgres).

def maybe_init_db():
    # Si existiera DATABASE_URL podrías conectar aquí.
    return None

def latest_run(conn=None):
    try:
        with open('/mnt/data/last_signals.json','r') as f:
            return json.load(f)
    except Exception:
        return None

def save_run(conn, payload):
    # Sin efecto si no hay DB; ya persistimos en /mnt/data desde app_evolutivo
    return True
