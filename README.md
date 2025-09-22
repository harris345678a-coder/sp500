# Evolutivo – Ranking diario (Top 50 → Top 3)

Servicio en FastAPI para seleccionar diariamente las 50 mejores oportunidades y, de ellas, las 3 más fuertes del día.

## Reglas de puntuación (clásicas y robustas)
- **Liquidez (50%)**: ADV20 en dólares (precio × volumen, promedio 20 días).
- **Volatilidad (30%)**: ATR(14) relativo al precio (ATR%).
- **Momentum (20%)**: magnitud del retorno de 5 días (|ret5|).

Las tres métricas se transforman a percentiles en el universo y se combinan.
Se ordena de mayor a menor para obtener el Top 50 y Top 3.

## Endpoints
- `POST /rank/run` → ejecuta el cálculo y guarda (Postgres si hay `DATABASE_URL`) y en `/mnt/data/last_rank.json`.
- `GET /rank/top50` y `GET /rank/top3` → devuelven el último resultado.
- `GET /healthz` → salud.

## Configuración en Render
- Build: `pip install -r requirements_evolutivo.txt`
- Start: `gunicorn -k gevent -w 1 app_evolutivo:app --bind 0.0.0.0:$PORT --timeout 180 --access-logfile /dev/null --error-logfile -`
- Cron (L-V 13:00 UTC): `python jobs/daily_rank.py`

## Dependencias sugeridas (añadir a tu requirements_evolutivo.txt)
fastapi
uvicorn
pandas
numpy
yfinance
psycopg2-binary
pyyaml

## Notas IBKR
Para datos en vivo desde Interactive Brokers se añadirá un backend con `ib_insync` y un Gateway accesible.
