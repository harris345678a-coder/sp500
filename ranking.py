import datetime as dt
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import yaml

from data_backend import make_backend, DataError

# --------- Métricas ----------

def atr(df: pd.DataFrame, period: int = 14) -> float:
    h, l, c = df['High'], df['Low'], df['Close']
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr_series = tr.rolling(period, min_periods=period).mean()
    val = float(atr_series.iloc[-1])
    if not np.isfinite(val):
        raise ValueError("ATR inválido")
    return val

def pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)

# --------- Universo ----------

def load_universe(path: str = "symbols.yaml") -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    # Validación mínima
    clean = []
    for item in data:
        ys = item.get("ysymbol") or item.get("code")
        if not ys:
            continue
        clean.append({"code": item.get("code", ys), "ysymbol": ys, "type": item.get("type")})
    if not clean:
        raise RuntimeError("Universo vacío o inválido")
    return clean

# --------- Ranking ----------

def compute_metrics_strict(ysymbol: str, start: dt.date, end: dt.date) -> Dict[str, float]:
    """
    Sin valores nulos ni suposiciones. Si no puede calcular métricas reales, lanza excepción.
    Requisitos:
    - Columnas: Open, High, Low, Close, Volume
    - Filas: >= 30
    - ADV20: requiere >= 20 filas con volumen disponible.
    - ATR%: requiere >= 15 filas.
    - Ret5: requiere >= 6 filas.
    """
    backend = make_backend()
    df = backend.history(ysymbol, start, end)

    # Validaciones de cantidad
    if df.shape[0] < 30:
        raise ValueError("Histórico insuficiente (<30)")
    if (df[['Open','High','Low','Close']].isna().any().any()):
        raise ValueError("Nulos en OHLC")
    if 'Volume' not in df.columns:
        raise ValueError("Sin columna Volume")

    # Liquidez: ADV20 USD
    last20 = df.tail(20)
    if last20.shape[0] < 20 or (last20['Volume'] <= 0).all():
        raise ValueError("Volumen insuficiente para ADV20")
    adv20 = float((last20['Close'] * last20['Volume']).mean())

    # ATR% 14
    if df.shape[0] < 15:
        raise ValueError("Histórico insuficiente para ATR")
    atr14 = atr(df, 14)
    last_close = float(df['Close'].iloc[-1])
    if last_close == 0.0:
        raise ValueError("Close final = 0")
    atrp = float(atr14 / last_close)

    # Momentum 5 días
    if df.shape[0] < 6:
        raise ValueError("Histórico insuficiente para ret5")
    ret5 = float(df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1.0)

    return {"adv20_usd": adv20, "atrp": atrp, "ret5": ret5}

def score_universe(universe: List[Dict[str, Any]], start: dt.date, end: dt.date) -> pd.DataFrame:
    rows = []
    skipped = []
    for u in universe:
        ys = u["ysymbol"]
        try:
            m = compute_metrics_strict(ys, start, end)
            rows.append({"code": u["code"], "ysymbol": ys, **m, "type": u.get("type")})
        except Exception as e:
            skipped.append({"code": u.get("code", ys), "ysymbol": ys, "reason": str(e)})
            continue
    if len(rows) == 0:
        raise RuntimeError("No se pudieron obtener datos reales para ningún símbolo")

    df = pd.DataFrame(rows)

    # Percentiles (sin NaN porque todo fue estrictamente validado)
    liq_p = pct_rank(np.log10(pd.to_numeric(df['adv20_usd'])))
    vol_p = pct_rank(pd.to_numeric(df['atrp']))
    mom_p = pct_rank(pd.to_numeric(abs(df['ret5'])))

    score = 0.50 * liq_p + 0.30 * vol_p + 0.20 * mom_p
    df['score'] = (score * 100.0).astype(float)

    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    df.attrs['skipped'] = skipped
    return df

def select_top(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.head(n).copy()

def run_daily_rank(universe_path: str = "symbols.yaml", lookback_days: int = 120) -> Dict[str, Any]:
    """
    Ejecuta el ranking estricto:
    - Cae cualquier símbolo que no cumpla condiciones (no se rellena nada).
    - Devuelve Top50 y Top3 con los símbolos válidos.
    - Incluye lista de 'skipped' en el payload para trazabilidad.
    """
    today = dt.date.today()
    start = today - dt.timedelta(days=lookback_days)
    uni = load_universe(universe_path)
    ranked = score_universe(uni, start, today)

    top50 = select_top(ranked, 50)
    top3  = select_top(ranked, 3)

    payload = {
        "as_of": today.isoformat(),
        "top50": top50.to_dict(orient="records"),
        "top3": top3.to_dict(orient="records"),
        "skipped": ranked.attrs.get('skipped', []),
        "universe_size": len(uni),
        "valid_count": len(ranked),
    }
    return payload
