import os
import json
import math
import datetime as dt
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import yaml

from data_backend import make_backend, DataError

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    roll_down = down.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df['High'], df['Low'], df['Close'])
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

def adx(df: pd.DataFrame, period: int = 14):
    up_move = df['High'].diff()
    down_move = df['Low'].diff() * -1
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    tr = true_range(df['High'], df['Low'], df['Close'])
    atr_ = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / (atr_ + 1e-12))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean() / (atr_ + 1e-12))
    dx = 100 * ( (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12) )
    adx_ = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return plus_di, minus_di, adx_

def mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0
    mf = tp * df['Volume']
    pos_mf = (tp > tp.shift(1)) * mf
    neg_mf = (tp < tp.shift(1)) * mf
    pos_sum = pos_mf.rolling(n, min_periods=n).sum()
    neg_sum = neg_mf.rolling(n, min_periods=n).sum().abs() + 1e-12
    mr = pos_sum / neg_sum
    return 100 - (100 / (1 + mr))

def rci(series: pd.Series, n: int = 14) -> pd.Series:
    if series.size < n:
        return pd.Series(index=series.index, dtype=float)
    idx = np.arange(n)
    out = np.full(series.shape, np.nan, dtype=float)
    values = series.values
    for i in range(n-1, len(series)):
        window = values[i-n+1:i+1]
        ranks_price = pd.Series(window).rank().values
        ranks_time = pd.Series(idx).rank().values
        r = np.corrcoef(ranks_price, ranks_time)[0,1]
        out[i] = r
    return pd.Series(out, index=series.index)

def pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)

def choppiness_metric(df_15m: pd.DataFrame, df_1d: pd.DataFrame) -> float:
    ret15 = df_15m['Close'].pct_change().dropna()
    std15 = ret15.rolling(20, min_periods=10).std().iloc[-1] if len(ret15) >= 10 else np.nan
    atrp = (atr(df_1d, 14).iloc[-1] / df_1d['Close'].iloc[-1]) if len(df_1d) >= 15 else np.nan
    if not np.isfinite(std15) or not np.isfinite(atrp) or atrp == 0:
        return np.nan
    return float(std15 / atrp)

def load_universe(path: str = "symbols.yaml") -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    clean = []
    for item in data:
        ys = item.get("ysymbol") or item.get("code")
        if not ys:
            continue
        clean.append({"code": item.get("code", ys), "ysymbol": ys, "type": item.get("type")})
    if not clean:
        raise RuntimeError("Universo vacío o inválido")
    return clean

def compute_top50(universe: List[Dict[str, Any]]) -> Dict[str, Any]:
    backend = make_backend()
    today = dt.datetime.utcnow()
    start_1d = today - dt.timedelta(days=200)
    start_15m = today - dt.timedelta(days=30)

    symbols = [u['ysymbol'] for u in universe]
    d1_map = backend.history_bulk(symbols, '1d', start_1d, today)
    d15_map = backend.history_bulk(symbols, '15m', start_15m, today)

    rows = []
    ema_ctx = {}
    for u in universe:
        s = u['ysymbol']
        df1 = d1_map.get(s)
        df15 = d15_map.get(s)
        if df1 is None or df15 is None or df1.empty or df15.empty:
            continue
        if len(df1) < 30:
            continue
        adv20 = float((df1['Close'] * df1['Volume']).rolling(20, min_periods=20).mean().iloc[-1])
        atrp = float(atr(df1, 14).iloc[-1] / df1['Close'].iloc[-1]) if len(df1) >= 15 else np.nan
        chop = choppiness_metric(df15, df1)
        if not np.isfinite(adv20) or not np.isfinite(atrp) or not np.isfinite(chop):
            continue
        ema20 = float( (df1['Close'].ewm(span=20, adjust=False, min_periods=20).mean()).iloc[-1] )
        ema50 = float( (df1['Close'].ewm(span=50, adjust=False, min_periods=50).mean()).iloc[-1] )
        ema200 = float( (df1['Close'].ewm(span=200, adjust=False, min_periods=200).mean()).iloc[-1] ) if len(df1) >= 200 else float('nan')
        close = float(df1['Close'].iloc[-1])
        bull = (close > ema20 > ema50) and (ema50 > (ema200 if np.isfinite(ema200) else ema50*0.99))
        bear = (close < ema20 < ema50) and (ema50 < (ema200 if np.isfinite(ema200) else ema50*1.01))
        bias = 1 if bull else (-1 if bear else 0)
        ema_ctx[s] = {"ema20": ema20, "ema50": ema50, "ema200": ema200, "bias": bias}

        rows.append({
            "code": u["code"], "ysymbol": s, "type": u.get("type"),
            "adv20_usd": adv20, "atrp": atrp, "chop": chop
        })

    if not rows:
        raise RuntimeError("Sin candidatos para Top 50")

    df = pd.DataFrame(rows)
    liq_p = pct_rank(np.log10(df['adv20_usd']))
    vol_p = pct_rank(df['atrp'])
    anti_chop_p = 1 - pct_rank(df['chop'])
    score = 0.50 * liq_p + 0.30 * vol_p + 0.20 * anti_chop_p
    bonus = df['ysymbol'].map(lambda s: 0.05 if ema_ctx[s]['bias'] == 1 else (-0.05 if ema_ctx[s]['bias'] == -1 else 0.0))
    score = (score + bonus).clip(lower=0, upper=1)
    df['score'] = (score * 100.0).astype(float)
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    top50 = df.head(50).copy()
    return {"top50": top50, "ema_ctx": ema_ctx, "d1_map": d1_map, "d15_map": d15_map}

def compute_top3(prev_ctx: Dict[str, Any]) -> Dict[str, Any]:
    backend = make_backend()
    top50 = prev_ctx["top50"]
    d1_map = prev_ctx["d1_map"]

    today = dt.datetime.utcnow()
    start_60m = today - dt.timedelta(days=30)
    start_1m = today - dt.timedelta(days=7)
    symbols = list(top50['ysymbol'])

    d60_map = backend.history_bulk(symbols, '60m', start_60m, today)
    try1m = backend.history_bulk(symbols, '1m', start_1m, today)
    missing_for_1m = [s for s in symbols if s not in try1m or try1m[s].empty]
    d1m_map = dict(try1m)
    if missing_for_1m:
        try5m = backend.history_bulk(missing_for_1m, '5m', start_1m, today)
        d1m_map.update(try5m)

    rows = []
    for _, row in top50.iterrows():
        s = row['ysymbol']
        df1 = d1_map.get(s)
        df60 = d60_map.get(s)
        df1m = d1m_map.get(s)
        if df1 is None or df60 is None or df1m is None or df1.empty or df60.empty or df1m.empty:
            continue

        rsi_15 = rsi(df1m['Close'], 14).iloc[-1] if len(df1m) >= 20 else np.nan
        rsi_60 = rsi(df60['Close'], 14).iloc[-1] if len(df60) >= 20 else np.nan
        rsi_1d = rsi(df1['Close'], 14).iloc[-1] if len(df1) >= 20 else np.nan

        plus_di, minus_di, adx_ = adx(df60, 14) if len(df60) >= 30 else (pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float))
        adx_last = adx_.iloc[-1] if not adx_.empty else np.nan
        plus_last = plus_di.iloc[-1] if not plus_di.empty else np.nan
        minus_last = minus_di.iloc[-1] if not minus_di.empty else np.nan

        rci_val = rci(df60['Close'], 14).iloc[-1] if len(df60) >= 30 else np.nan
        mfi_val = mfi(df60, 14).iloc[-1] if len(df60) >= 20 else np.nan

        ob_score = np.nan  # En Yahoo no hay order book; si activas backend IBKR, cámbialo allí.
        require_ob = os.getenv("ORDERBOOK_REQUIRED", "0") == "1"
        ob_pass = (not require_ob)  # si se requiere y no hay book -> no pasa

        adx_ok = np.isfinite(adx_last) and adx_last >= float(os.getenv("THRESH_ADX_MIN", "20"))
        rsi_align_long = all([np.isfinite(x) for x in [rsi_15, rsi_60]]) and (rsi_60 > 50 and 45 <= rsi_15 <= 70)
        rsi_align_short = all([np.isfinite(x) for x in [rsi_15, rsi_60]]) and (rsi_60 < 50 and 30 <= rsi_15 <= 55)
        rci_ok = np.isfinite(rci_val) and abs(rci_val) >= float(os.getenv("THRESH_RCI_ABS_MIN", "0.3"))
        mfi_ok = np.isfinite(mfi_val) and (40 <= mfi_val <= 80)

        parts = []
        if np.isfinite(adx_last): parts.append(min(adx_last/50, 1.0))
        if np.isfinite(rci_val): parts.append(min(abs(rci_val), 1.0))
        if np.isfinite(mfi_val): parts.append(1-abs((mfi_val-50)/50))
        if np.isfinite(rsi_15) and np.isfinite(rsi_60):
            rsi_score = 0.5*(1-abs((rsi_15-55)/55)) + 0.5*(1-abs((rsi_60-55)/55))
            parts.append(max(rsi_score, 0.0))
        comp = float(np.mean(parts)) if parts else 0.0

        rows.append({
            "ysymbol": s,
            "code": row['code'],
            "type": row['type'],
            "score50": row['score'],
            "score12": round(comp*100, 2),
            "adx": float(adx_last) if np.isfinite(adx_last) else None,
            "plus_di": float(plus_last) if np.isfinite(plus_last) else None,
            "minus_di": float(minus_last) if np.isfinite(minus_last) else None,
            "rsi15": float(rsi_15) if np.isfinite(rsi_15) else None,
            "rsi60": float(rsi_60) if np.isfinite(rsi_60) else None,
            "rsi1d": float(rsi_1d) if np.isfinite(rsi_1d) else None,
            "rci": float(rci_val) if np.isfinite(rci_val) else None,
            "mfi": float(mfi_val) if np.isfinite(mfi_val) else None,
            "ob_score": float(ob_score) if np.isfinite(ob_score) else None,
            "ob_pass": ob_pass,
            "adx_ok": adx_ok,
            "rsi_align_long": rsi_align_long,
            "rsi_align_short": rsi_align_short,
            "rci_ok": rci_ok,
            "mfi_ok": mfi_ok
        })

    if not rows:
        raise RuntimeError("No se pudo calcular Top3 (faltan TF o indicadores)")

    df = pd.DataFrame(rows)
    keep_mask = (
        df['adx_ok'].fillna(False) &
        df['rci_ok'].fillna(False) &
        df['mfi_ok'].fillna(False) &
        df['ob_pass'].fillna(True)
    )
    kept = df[keep_mask].copy()
    if kept.empty:
        kept = df.copy()

    kept['rank_score'] = (
        0.30 * (kept['adx'].fillna(0)/50).clip(0,1) +
        0.20 * (kept['rci'].abs().fillna(0)).clip(0,1) +
        0.20 * (1-((kept['mfi']-50).abs()/50)).clip(0,1) +
        0.20 * ( ( (1-abs((kept['rsi15']-55)/55)) + (1-abs((kept['rsi60']-55)/55)) )/2 ).fillna(0).clip(0,1) +
        0.10 * kept['score50'].fillna(0)/100.0
    )
    kept = kept.sort_values('rank_score', ascending=False).reset_index(drop=True)
    top3 = kept.head(3).copy()
    return {"top3_raw": df, "top3": top3}

def swing_levels(series: pd.Series, window: int = 20):
    hi = float(series.rolling(window, min_periods=window).max().iloc[-1])
    lo = float(series.rolling(window, min_periods=window).min().iloc[-1])
    return lo, hi

def build_signal_for_symbol(symbol: str, d1: pd.DataFrame, d60: pd.DataFrame, d1m: pd.DataFrame,
                            feat: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    adx_v = feat.get("adx")
    rsi15 = feat.get("rsi15")
    rsi60 = feat.get("rsi60")
    rci_v = feat.get("rci")
    mfi_v = feat.get("mfi")
    close = float(d1['Close'].iloc[-1])

    side = None
    strategy = None

    if rsi60 and rsi60 > 50 and rci_v and rci_v > 0.3 and adx_v and adx_v >= 20:
        side = "long"
        lo, hi = swing_levels(d1['Close'], 20)
        trigger = max(hi, close)
        atr14 = float(atr(d1, 14).iloc[-1])
        sl = trigger - 1.0 * atr14
        tp = trigger + float(os.getenv("TP_ATR_MULT", "1.8")) * atr14
        strategy = "breakout" if trigger == hi else "momentum"
    elif rsi60 and rsi60 < 50 and rci_v and rci_v < -0.3 and adx_v and adx_v >= 20:
        side = "short"
        lo, hi = swing_levels(d1['Close'], 20)
        trigger = min(lo, close)
        atr14 = float(atr(d1, 14).iloc[-1])
        sl = trigger + 1.0 * atr14
        tp = trigger - float(os.getenv("TP_ATR_MULT", "1.8")) * atr14
        strategy = "breakout" if trigger == lo else "momentum"
    else:
        rsi1d = float(rsi(d1['Close'], 14).iloc[-1]) if len(d1) >= 20 else np.nan
        atr14 = float(atr(d1, 14).iloc[-1]) if len(d1) >= 15 else np.nan
        if np.isfinite(rsi1d) and np.isfinite(atr14):
            if rsi1d < 30:
                side = "long"
                trigger = close + 0.2 * atr14
                sl = close - 0.8 * atr14
                tp = close + 1.6 * atr14
                strategy = "reversion"
            elif rsi1d > 70:
                side = "short"
                trigger = close - 0.2 * atr14
                sl = close + 0.8 * atr14
                tp = close - 1.6 * atr14
                strategy = "reversion"
    if side is None:
        return None

    rr = abs((tp - trigger) / (trigger - sl)) if (trigger != sl) else 0.0
    if rr < float(os.getenv("THRESH_RR_MIN", "1.8")):
        return None
    anch = abs(close - trigger) / (float(atr(d1, 14).iloc[-1]) + 1e-9)
    if anch > float(os.getenv("THRESH_ANCHOR_ATR", "0.25")):
        return None

    lo20, hi20 = swing_levels(d1['Close'], 20)
    rng = hi20 - lo20
    fibo = {
        "38.2": hi20 - 0.382 * rng,
        "50.0": hi20 - 0.500 * rng,
        "61.8": hi20 - 0.618 * rng
    }

    signal = {
        "symbol": symbol,
        "side": side,
        "strategy": strategy,
        "trigger": round(float(trigger), 4),
        "sl": round(float(sl), 4),
        "tp": round(float(tp), 4),
        "rr": round(float(rr), 2),
        "rsi15": feat.get("rsi15"),
        "rsi60": feat.get("rsi60"),
        "adx": feat.get("adx"),
        "rci": feat.get("rci"),
        "mfi": feat.get("mfi"),
        "ob_score": feat.get("ob_score"),
        "fibo": fibo
    }
    return signal

def run_full_pipeline(universe_path: str = "symbols.yaml") -> Dict[str, Any]:
    uni = load_universe(universe_path)
    phase1 = compute_top50(uni)
    phase2 = compute_top3(phase1)
    top3 = phase2["top3"]

    backend = make_backend()
    today = dt.datetime.utcnow()
    start_1d = today - dt.timedelta(days=200)
    start_60m = today - dt.timedelta(days=30)
    start_1m = today - dt.timedelta(days=7)

    approved = []
    reasons = []
    for _, r in top3.iterrows():
        s = r['ysymbol']
        d1 = phase1["d1_map"][s]
        d60 = backend.history(s, '60m', start_60m, today)
        d1m = backend.history(s, '1m', start_1m, today)
        if d1m is None or d1m.empty:
            d1m = backend.history(s, '5m', start_1m, today)
        feat = {k: r.get(k) for k in ['adx','rsi15','rsi60','rci','mfi','ob_score']}
        sig = build_signal_for_symbol(s, d1, d60, d1m, feat)
        if sig:
            approved.append(sig)
        else:
            reasons.append({"symbol": s, "reason": "failed presend gates (RR/anchoring/strategy)"})
    return {
        "as_of": dt.date.today().isoformat(),
        "top50": phase1["top50"].to_dict(orient="records"),
        "top3_factors": phase2["top3"].to_dict(orient="records"),
        "approved_top3": approved,
        "rejected_top3": reasons
    }
