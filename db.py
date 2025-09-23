import os, json

# Directorio de datos: por defecto /tmp (writable en Render Free)
DATA_DIR = os.getenv("DATA_DIR", "/tmp/evolutivo")
LAST_JSON = os.path.join(DATA_DIR, "last_signals.json")

def maybe_init_db():
    # Sin Postgres: devolver None (stubs seguros)
    return None

def latest_run(conn=None):
    try:
        with open(LAST_JSON, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def save_run(conn, payload):
    # Ya persistimos en el FS por app_evolutivo; aquí no hace falta nada.
    return True
