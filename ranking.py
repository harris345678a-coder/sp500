import os, math, json, datetime as dt
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import yaml

from data_backend import make_backend

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df['High'], df['Low'], df['Close']
    prev_c = c.shift(1)
    tr = pd.concat([h-l, (h-prev_c).abs(), (l-prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)

def load_universe(path: str = "symbols.yaml") -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def compute_metrics_for_symbol(ysymbol: str, start: dt.date, end: dt.date) -> Dict[str, float]:
    backend = make_backend()
    df = backend.history(ysymbol, start, end)
    if df is None or df.empty:
        return {"adv20_usd": np.nan, "atrp": np.nan, "ret5": np.nan}

    df = df.dropna(subset=['Close']).copy()
    if df.empty:
        return {"adv20_usd": np.nan, "atrp": np.nan, "ret5": np.nan}

    if 'Volume' in df and (df['Volume'] > 0).any():
        adv20 = (df['Close'] * df['Volume']).rolling(20).mean().iloc[-1]
    else:
        adv20 = np.nan

    atr14 = atr(df, 14).iloc[-1] if len(df) >= 15 else np.nan
    close = df['Close'].iloc[-1]
    atrp = (atr14 / close) if (atr14 is not None and close and not np.isnan(atr14)) else np.nan

    ret5 = (df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1.0) if len(df) >= 6 else np.nan

    return {"adv20_usd": float(adv20) if adv20==adv20 else np.nan,
            "atrp": float(atrp) if atrp==atrp else np.nan,
            "ret5": float(ret5) if ret5==ret5 else np.nan}

def score_universe(universe: List[Dict[str, Any]], start: dt.date, end: dt.date) -> pd.DataFrame:
    rows = []
    for u in universe:
        ys = u.get("ysymbol", u["code"])
        m = compute_metrics_for_symbol(ys, start, end)
        rows.append({"code": u["code"], "ysymbol": ys, **m, "type": u["type"]})
    df = pd.DataFrame(rows)

    liq_p = pct_rank(np.log10(df['adv20_usd'].replace({0:np.nan}))).fillna(0.0)
    vol_p = pct_rank(df['atrp']).fillna(0.0)
    mom_p = pct_rank(np.abs(df['ret5'])).fillna(0.0)

    score = 0.50 * liq_p + 0.30 * vol_p + 0.20 * mom_p
    df['score'] = score * 100.0

    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    return df

def select_top(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.head(n).copy()

def run_daily_rank(universe_path: str = "symbols.yaml", lookback_days: int = 90) -> Dict[str, Any]:
    today = dt.date.today()
    start = today - dt.timedelta(days=lookback_days)
    uni = load_universe(universe_path)
    ranked = score_universe(uni, start, today)

    top50 = select_top(ranked, 50)
    top3  = select_top(ranked, 3)

    payload = {
        "as_of": today.isoformat(),
        "top50": top50.to_dict(orient="records"),
        "top3": top3.to_dict(orient="records")
    }
    return payload
