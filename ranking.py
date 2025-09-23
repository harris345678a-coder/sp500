import os
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from data_backend import DataBackend, DataError

# ------------------ Indicators ------------------

def rsi(series: pd.Series, n: int = 14) -> float:
    s = series.dropna().astype(float)
    if s.size < n + 2:
        return float("nan")
    delta = s.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=s.index).ewm(alpha=1/n, adjust=False).mean()
    roll_down = pd.Series(loss, index=s.index).ewm(alpha=1/n, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    rsi_val = 100 - (100 / (1 + rs))
    return float(rsi_val.iloc[-1])

def _dmi(df: pd.DataFrame, n: int = 14) -> Tuple[float,float,float]:
    # returns (ADX, +DI, -DI)
    h, l, c = df["High"], df["Low"], df["Close"]
    up = h.diff()
    down = -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    atr = pd.Series(tr, index=df.index).rolling(n, min_periods=n).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(n, min_periods=n).mean() / (atr + 1e-12))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(n, min_periods=n).mean() / (atr + 1e-12))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)) * 100
    adx = dx.rolling(n, min_periods=n).mean()
    return float(adx.dropna().iloc[-1]) if not adx.dropna().empty else float("nan"),                float(plus_di.dropna().iloc[-1]) if not plus_di.dropna().empty else float("nan"),                float(minus_di.dropna().iloc[-1]) if not minus_di.dropna().empty else float("nan")

def mfi(df: pd.DataFrame, n: int = 14) -> float:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
    rmf = tp * df.get("Volume", 0).fillna(0)
    pos = np.where(tp.diff() > 0, rmf, 0.0)
    neg = np.where(tp.diff() < 0, rmf, 0.0)
    pos_n = pd.Series(pos, index=df.index).rolling(n, min_periods=n).sum()
    neg_n = pd.Series(neg, index=df.index).rolling(n, min_periods=n).sum()
    mr = pos_n / (neg_n + 1e-12)
    mfi_val = 100 - (100 / (1 + mr))
    mfi_val = mfi_val.dropna()
    return float(mfi_val.iloc[-1]) if not mfi_val.empty else float("nan")

def rci(series: pd.Series, n: int = 9) -> float:
    # Rank Correlation Index
    s = series.dropna().astype(float)
    if s.size < n:
        return float("nan")
    s = s.iloc[-n:]
    price_rank = pd.Series(s, index=s.index).rank(method="first")
    time_rank = pd.Series(range(1, n+1), index=s.index)
    diff = price_rank - time_rank
    d2 = (diff ** 2).sum()
    rci_val = (1 - (6 * d2) / (n * (n**2 - 1))) * 100
    return float(rci_val)

def atrp(df: pd.DataFrame, n: int = 14) -> float:
    # ATR% of price (normalized volatility)
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = np.maximum(h - l, np.maximum((h - c.shift()).abs(), (l - c.shift()).abs()))
    atr = pd.Series(tr, index=df.index).rolling(n, min_periods=n).mean()
    px = c.rolling(n, min_periods=n).mean()
    val = (atr / (px + 1e-12)).dropna()
    return float(val.iloc[-1]) if not val.empty else float("nan")

def choppiness(df: pd.DataFrame, n: int = 14) -> float:
    # Simplified choppiness proxy: std of returns
    ret = df["Close"].pct_change()
    val = ret.rolling(n, min_periods=n).std().dropna()
    return float(val.iloc[-1]) if not val.empty else float("nan")

# ------------------ Universe ------------------

def _default_universe() -> List[Dict]:
    syms = [
        "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","BRK-B","JPM","XOM",
        "SPY","QQQ","DIA","IWM","IVV","VOO","VTI","SOXX","SMH","HYG","LQD","TLT",
        "GLD","SLV","GDX","USO","XLE","XLF","XLK","XLV","XLY","XLI","XLP","XLU","XLB",
        "EFA","EEM","IYR","KRE","IBB","XHB","XOP","DBA","DBC","IEF","SHY","VIXY","BITO"
    ]
    out = []
    for s in syms:
        out.append({"code": s.replace("-", ""), "ysymbol": s, "type": "equity" if s.isalpha() and len(s) <= 5 else "etf"})
    return out

def load_universe() -> List[Dict]:
    import yaml
    path = os.getenv("SYMBOLS_PATH", "symbols.yaml")
    try:
        with open(path, "r") as f:
            y = yaml.safe_load(f) or {}
        lst = y.get("symbols") or []
        if lst:
            out = []
            for it in lst:
                if isinstance(it, dict):
                    code = it.get("code") or it.get("symbol") or it.get("ysymbol")
                    ysym = it.get("ysymbol") or code
                    typ = it.get("type") or ("etf" if code and code.isupper() and len(code) > 4 else "equity")
                    out.append({"code": code, "ysymbol": ysym, "type": typ})
                elif isinstance(it, str):
                    out.append({"code": it.replace("-", ""), "ysymbol": it, "type": "equity"})
            return out
    except Exception:
        pass
    return _default_universe()

# ------------------ Core Metrics ------------------

def compute_liquidity_volatility(backend: DataBackend, symbol: str) -> Dict:
    d1 = backend.history(symbol, "1d")
    if d1 is None or d1.empty:
        return {"adv20_usd": float("nan"), "atrp": float("nan"), "chop": float("nan")}
    adv20 = (d1["Close"] * d1.get("Volume", 0)).rolling(20, min_periods=10).mean().iloc[-1]
    return {
        "adv20_usd": float(adv20) if pd.notna(adv20) else float("nan"),
        "atrp": atrp(d1, 14),
        "chop": choppiness(d1, 20),
    }

def compute_timeframe_factors(backend: DataBackend, symbol: str) -> Dict:
    d15 = backend.history(symbol, "15m")
    d60 = backend.history(symbol, "60m")
    d240 = backend.history(symbol, "240m")
    d1 = backend.history(symbol, "1d")

    facts = {}
    if d15 is not None and not d15.empty:
        facts["rsi15"] = rsi(d15["Close"], 14)
    if d60 is not None and not d60.empty:
        a, pdi, mdi = _dmi(d60, 14); facts["adx"] = a; facts["plus_di"] = pdi; facts["minus_di"] = mdi
        facts["rsi60"] = rsi(d60["Close"], 14)
    if d240 is not None and not d240.empty:
        a4, pdi4, mdi4 = _dmi(d240, 14); facts["adx4h"] = a4; facts["plus_di4h"] = pdi4; facts["minus_di4h"] = mdi4
        facts["rsi4h"] = rsi(d240["Close"], 14)
        facts["rci4h"] = rci(d240["Close"], 9)
        facts["mfi4h"] = mfi(d240, 14)
    if d1 is not None and not d1.empty:
        facts["rsi1d"] = rsi(d1["Close"], 14)

    rl = (facts.get("rsi15", 50) <= 70) and (facts.get("rsi60", 50) >= 55) and (facts.get("rsi4h", 50) >= 55) and (facts.get("rsi1d", 50) >= 55)
    rs = (facts.get("rsi15", 50) >= 30) and (facts.get("rsi60", 50) <= 45) and (facts.get("rsi4h", 50) <= 45) and (facts.get("rsi1d", 50) <= 45)
    facts["rsi_align_long"] = bool(rl)
    facts["rsi_align_short"] = bool(rs)

    facts["rci"] = facts.get("rci4h", float("nan"))
    facts["mfi"] = facts.get("mfi4h", float("nan"))

    return facts

def score_symbol_for_top50(liqvol: Dict) -> float:
    adv = liqvol.get("adv20_usd", float("nan"))
    atrp_v = liqvol.get("atrp", float("nan"))
    chop = liqvol.get("chop", float("nan"))
    s_adv = np.tanh((adv or 0.0) / 1e10)
    s_vol = np.tanh((atrp_v or 0.0) * 10)
    s_chop = 1.0 - np.tanh((chop or 0.0) * 10)
    score = (0.5 * s_adv) + (0.4 * s_vol) + (0.1 * s_chop)
    return float(np.clip(score * 100, 0, 100))

def compute_top50(universe: List[Dict]) -> List[Dict]:
    backend = DataBackend()
    rows = []
    for it in universe:
        sym = it["ysymbol"]
        lv = compute_liquidity_volatility(backend, sym)
        score = score_symbol_for_top50(lv)
        row = {
            "code": it["code"],
            "ysymbol": sym,
            "type": it.get("type", "equity"),
            "adv20_usd": lv["adv20_usd"],
            "atrp": lv["atrp"],
            "chop": lv["chop"],
            "score": score,
        }
        rows.append(row)

    rows = sorted(rows, key=lambda r: r["score"], reverse=True)[:50]
    return rows

def compute_top3_factors(top50: List[Dict]) -> List[Dict]:
    backend = DataBackend()
    cands = [r["ysymbol"] for r in top50[:12]]
    out = []
    for sym in cands:
        facts = compute_timeframe_factors(backend, sym)
        base = next((r["score"] for r in top50 if r["ysymbol"] == sym), 0.0)
        align_boost = 0.0
        if facts.get("rsi_align_long"): align_boost += 8.0
        if facts.get("rsi_align_short"): align_boost += 8.0
        if (facts.get("adx4h", 0) or 0) >= 25: align_boost += 4.0
        if (facts.get("rci4h", 0) or 0) >= 0.9: align_boost += 3.0
        if (facts.get("mfi4h", 0) or 0) >= 60: align_boost += 3.0
        rank_score = float(np.clip(base + align_boost, 0, 100))
        out.append({
            "ysymbol": sym,
            "code": sym.replace("-", ""),
            "type": "equity" if sym.isalpha() and len(sym) <= 5 else "etf",
            "score50": base,
            "score12": rank_score,
            "adx": facts.get("adx"),
            "plus_di": facts.get("plus_di"),
            "minus_di": facts.get("minus_di"),
            "rsi15": facts.get("rsi15"),
            "rsi60": facts.get("rsi60"),
            "rsi4h": facts.get("rsi4h"),
            "rsi1d": facts.get("rsi1d"),
            "rci": facts.get("rci"),
            "mfi": facts.get("mfi"),
            "adx4h": facts.get("adx4h"),
            "rci4h": facts.get("rci4h"),
            "mfi4h": facts.get("mfi4h"),
            "rsi_align_long": facts.get("rsi_align_long"),
            "rsi_align_short": facts.get("rsi_align_short"),
            "rank_score": rank_score,
            "ob_score": None,
            "ob_pass": True,
            "adx_ok": (facts.get("adx", 0) or 0) >= 20,
            "rsi_align_long_ok": bool(facts.get("rsi_align_long")),
            "rsi_align_short_ok": bool(facts.get("rsi_align_short")),
            "rci_ok": (facts.get("rci", 0) or 0) >= 0.85,
            "mfi_ok": (facts.get("mfi", 0) or 0) >= 55,
        })
    out = sorted(out, key=lambda x: x["rank_score"], reverse=True)[:3]
    return out

def run_full_pipeline() -> Dict:
    uni = load_universe()
    top50 = compute_top50(uni)
    top3_factors = compute_top3_factors(top50)
    payload: Dict = {
        "as_of": datetime.utcnow().date().isoformat(),
        "top50": top50,
        "top3_factors": top3_factors,
        "approved_top3": [],
        "rejected_top3": [],
    }
    return payload
