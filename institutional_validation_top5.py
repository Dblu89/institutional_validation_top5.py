import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# =========================================================
# CONFIG DAS 5 MELHORES
# SUBSTITUA PELOS PARÂMETROS REAIS DAS SUAS 5 ESTRATÉGIAS
# =========================================================
TOP5_CONFIGS: List[Dict] = [
    {
        "name": "candidate_201",
        "params": {
            "use_session": False,
            "session_start": 10,
            "session_end": 16,
            "ema_fast": 13,
            "ema_slow": 34,
            "slippage": 0,
            "global_max_hold_bars": 7,
            "bo_lookback": 13,
            "bo_buffer": 1,
            "bo_min_range": 10,
            "bo_min_body": 1,
            "bo_min_atr_ratio": 0.49,
            "bo_expansion_body_atr": 0.25,
            "bo_vol_mult": 0.85,
            "bo_slope_min": 0.5231,
            "bo_require_trend": True,
            "bo_stop_points": 69,
            "bo_rr": 4.68,
            "breakout_min_score": 0.15,
            "sw_sweep_buffer": 4,
            "sw_rejection_body_atr": 0.49,
            "sw_stop_buffer": 1,
            "sw_rr": 2.78,
            "sweep_min_score": 3.67,
            "smc_bos_lookback": 5,
            "smc_bos_buffer": 7,
            "smc_displacement_body_atr": 0.84,
            "smc_pullback_atr": 0.16,
            "smc_stop_buffer": 6,
            "smc_rr": 4.09,
            "smc_min_score": 4.52,
        },
    },
    {
        "name": "candidate_625",
        "params": {
            "use_session": False,
            "session_start": 12,
            "session_end": 14,
            "ema_fast": 5,
            "ema_slow": 21,
            "slippage": 1,
            "global_max_hold_bars": 3,
            "bo_lookback": 5,
            "bo_buffer": 5,
            "bo_min_range": 17,
            "bo_min_body": 6,
            "bo_min_atr_ratio": 0.74,
            "bo_expansion_body_atr": 0.64,
            "bo_vol_mult": 0.81,
            "bo_slope_min": 0.8387,
            "bo_require_trend": True,
            "bo_stop_points": 44,
            "bo_rr": 3.58,
            "breakout_min_score": 0.62,
            "sw_sweep_buffer": 8,
            "sw_rejection_body_atr": 0.09,
            "sw_stop_buffer": 9,
            "sw_rr": 2.85,
            "sweep_min_score": 1.12,
            "smc_bos_lookback": 8,
            "smc_bos_buffer": 4,
            "smc_displacement_body_atr": 1.11,
            "smc_pullback_atr": 1.03,
            "smc_stop_buffer": 4,
            "smc_rr": 1.14,
            "smc_min_score": 4.87,
        },
    },
    {
        "name": "candidate_896",
        "params": {
            "use_session": False,
            "session_start": 10,
            "session_end": 15,
            "ema_fast": 13,
            "ema_slow": 34,
            "slippage": 1,
            "global_max_hold_bars": 38,
            "bo_lookback": 5,
            "bo_buffer": 17,
            "bo_min_range": 19,
            "bo_min_body": 19,
            "bo_min_atr_ratio": 1.04,
            "bo_expansion_body_atr": 0.84,
            "bo_vol_mult": 0.60,
            "bo_slope_min": 0.6361,
            "bo_require_trend": True,
            "bo_stop_points": 94,
            "bo_rr": 3.53,
            "breakout_min_score": 3.04,
            "sw_sweep_buffer": 2,
            "sw_rejection_body_atr": 0.42,
            "sw_stop_buffer": 17,
            "sw_rr": 1.07,
            "sweep_min_score": 3.51,
            "smc_bos_lookback": 8,
            "smc_bos_buffer": 15,
            "smc_displacement_body_atr": 0.23,
            "smc_pullback_atr": 0.51,
            "smc_stop_buffer": 1,
            "smc_rr": 3.29,
            "smc_min_score": 0.68,
        },
    },
    {
        "name": "candidate_1211",
        "params": {
            "use_session": False,
            "session_start": 10,
            "session_end": 13,
            "ema_fast": 9,
            "ema_slow": 34,
            "slippage": 2,
            "global_max_hold_bars": 4,
            "bo_lookback": 8,
            "bo_buffer": 7,
            "bo_min_range": 5,
            "bo_min_body": 16,
            "bo_min_atr_ratio": 0.06,
            "bo_expansion_body_atr": 0.70,
            "bo_vol_mult": 1.22,
            "bo_slope_min": 0.5650,
            "bo_require_trend": False,
            "bo_stop_points": 16,
            "bo_rr": 1.39,
            "breakout_min_score": 1.08,
            "sw_sweep_buffer": 9,
            "sw_rejection_body_atr": 0.93,
            "sw_stop_buffer": 7,
            "sw_rr": 1.16,
            "sweep_min_score": 3.63,
            "smc_bos_lookback": 5,
            "smc_bos_buffer": 14,
            "smc_displacement_body_atr": 0.58,
            "smc_pullback_atr": 0.80,
            "smc_stop_buffer": 6,
            "smc_rr": 2.03,
            "smc_min_score": 1.64,
        },
    },
    {
        "name": "candidate_1157",
        "params": {
            "use_session": True,
            "session_start": 9,
            "session_end": 17,
            "ema_fast": 13,
            "ema_slow": 34,
            "slippage": 0,
            "global_max_hold_bars": 59,
            "bo_lookback": 13,
            "bo_buffer": 3,
            "bo_min_range": 12,
            "bo_min_body": 6,
            "bo_min_atr_ratio": 0.40,
            "bo_expansion_body_atr": 0.09,
            "bo_vol_mult": 0.82,
            "bo_slope_min": 0.7381,
            "bo_require_trend": True,
            "bo_stop_points": 10,
            "bo_rr": 4.63,
            "breakout_min_score": 0.66,
            "sw_sweep_buffer": 3,
            "sw_rejection_body_atr": 0.49,
            "sw_stop_buffer": 13,
            "sw_rr": 1.08,
            "sweep_min_score": 2.06,
            "smc_bos_lookback": 13,
            "smc_bos_buffer": 13,
            "smc_displacement_body_atr": 0.30,
            "smc_pullback_atr": 1.00,
            "smc_stop_buffer": 17,
            "smc_rr": 4.61,
            "smc_min_score": 4.54,
        },
    },
]


# =========================================================
# LOAD / FEATURES
# =========================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower_map = {c.lower().strip(): c for c in df.columns}
    rename_map = {}

    aliases = {
        "open": ["open", "abertura"],
        "high": ["high", "max", "maximum", "alta"],
        "low": ["low", "min", "minimum", "baixa"],
        "close": ["close", "fechamento"],
        "volume": ["volume", "tick_volume", "vol"],
        "time": ["time", "datetime", "date", "timestamp"],
    }

    for target, candidates in aliases.items():
        for c in candidates:
            if c in lower_map:
                rename_map[lower_map[c]] = target
                break

    df = df.rename(columns=rename_map)

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV sem colunas obrigatórias: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    return df.reset_index(drop=True)


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV não encontrado: {path}")
    df = pd.read_csv(path)
    return normalize_columns(df)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["range"] = out["high"] - out["low"]
    out["body"] = (out["close"] - out["open"]).abs()
    out["body_dir"] = np.sign(out["close"] - out["open"])

    prev_close = out["close"].shift(1)
    tr = np.maximum(
        out["high"] - out["low"],
        np.maximum((out["high"] - prev_close).abs(), (out["low"] - prev_close).abs()),
    )
    out["tr"] = tr
    out["atr_14"] = out["tr"].rolling(14, min_periods=1).mean()
    out["atr_50"] = out["tr"].rolling(50, min_periods=1).mean()
    out["atr_ratio"] = (
        (out["atr_14"] / out["atr_50"].replace(0, np.nan))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
    )

    for span in [5, 9, 13, 21, 34, 50, 100]:
        out[f"ema_{span}"] = out["close"].ewm(span=span, adjust=False).mean()

    out["ema_21_slope"] = out["ema_21"].diff().fillna(0.0)
    out["ema_50_slope"] = out["ema_50"].diff().fillna(0.0)

    for lb in [3, 5, 8, 13, 21, 34]:
        out[f"hh_{lb}"] = out["high"].rolling(lb, min_periods=1).max().shift(1)
        out[f"ll_{lb}"] = out["low"].rolling(lb, min_periods=1).min().shift(1)

    out["vol_ma_20"] = out["volume"].rolling(20, min_periods=1).mean()
    out["recent_high_10"] = out["high"].rolling(10, min_periods=1).max().shift(1)
    out["recent_low_10"] = out["low"].rolling(10, min_periods=1).min().shift(1)

    return out


def build_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    arr = {}
    cols = [
        "open", "high", "low", "close", "volume",
        "range", "body", "body_dir", "tr",
        "atr_14", "atr_50", "atr_ratio",
        "ema_5", "ema_9", "ema_13", "ema_21", "ema_34", "ema_50", "ema_100",
        "ema_21_slope", "ema_50_slope",
        "vol_ma_20", "recent_high_10", "recent_low_10",
        "hh_3", "hh_5", "hh_8", "hh_13", "hh_21", "hh_34",
        "ll_3", "ll_5", "ll_8", "ll_13", "ll_21", "ll_34",
    ]
    for c in cols:
        arr[c] = df[c].astype(float).to_numpy()

    if "time" in df.columns:
        arr["time"] = df["time"].to_numpy()
        arr["hour"] = pd.to_datetime(df["time"]).dt.hour.fillna(12).astype(int).to_numpy()
        arr["month"] = pd.to_datetime(df["time"]).dt.to_period("M").astype(str).to_numpy()
        arr["date"] = pd.to_datetime(df["time"]).dt.date.astype(str).to_numpy()
    else:
        n = len(df)
        arr["time"] = np.array([None] * n)
        arr["hour"] = np.full(n, 12, dtype=int)
        arr["month"] = np.array(["unknown"] * n)
        arr["date"] = np.array(["unknown"] * n)

    return arr


# =========================================================
# CORE LOGIC
# =========================================================
def in_session(hour: int, start_hour: int, end_hour: int) -> bool:
    return start_hour <= hour <= end_hour


def calc_breakout_signal(arr: Dict[str, np.ndarray], i: int, p: Dict):
    if i < p["bo_lookback"] + 2:
        return None

    hour = int(arr["hour"][i])
    if p["use_session"] and not in_session(hour, p["session_start"], p["session_end"]):
        return None

    prev = i - 1
    if arr["range"][prev] < p["bo_min_range"]:
        return None
    if arr["body"][prev] < p["bo_min_body"]:
        return None
    if arr["atr_ratio"][prev] < p["bo_min_atr_ratio"]:
        return None

    fast = arr[f"ema_{p['ema_fast']}"][prev]
    slow = arr[f"ema_{p['ema_slow']}"][prev]
    slope = arr["ema_21_slope"][prev]
    hh = arr[f"hh_{p['bo_lookback']}"][prev]
    ll = arr[f"ll_{p['bo_lookback']}"][prev]

    score = 0.0
    if fast > slow:
        score += 1.5
    elif fast < slow:
        score += 1.5
    if abs(slope) >= p["bo_slope_min"]:
        score += 1.0
    if arr["body"][prev] >= arr["atr_14"][prev] * p["bo_expansion_body_atr"]:
        score += 2.0
    if arr["volume"][prev] >= arr["vol_ma_20"][prev] * p["bo_vol_mult"]:
        score += 0.5

    if arr["high"][i] >= hh + p["bo_buffer"]:
        if p["bo_require_trend"] and not (fast > slow):
            return None
        entry = hh + p["bo_buffer"] + p["slippage"]
        stop = entry - p["bo_stop_points"]
        target = entry + p["bo_stop_points"] * p["bo_rr"]
        return ("breakout", 1, score, entry, stop, target, "breakout_long")

    if arr["low"][i] <= ll - p["bo_buffer"]:
        if p["bo_require_trend"] and not (fast < slow):
            return None
        entry = ll - p["bo_buffer"] - p["slippage"]
        stop = entry + p["bo_stop_points"]
        target = entry - p["bo_stop_points"] * p["bo_rr"]
        return ("breakout", -1, score, entry, stop, target, "breakout_short")

    return None


def calc_sweep_signal(arr: Dict[str, np.ndarray], i: int, p: Dict):
    if i < 12:
        return None

    hour = int(arr["hour"][i])
    if p["use_session"] and not in_session(hour, p["session_start"], p["session_end"]):
        return None

    recent_high = arr["recent_high_10"][i]
    recent_low = arr["recent_low_10"][i]
    atr = arr["atr_14"][i]
    if np.isnan(recent_high) or np.isnan(recent_low) or np.isnan(atr):
        return None

    score = 0.0

    if arr["high"][i] > recent_high + p["sw_sweep_buffer"] and arr["close"][i] < recent_high:
        score += 2.0
        if arr["body"][i] >= atr * p["sw_rejection_body_atr"]:
            score += 1.5
        if arr["close"][i] < arr["open"][i]:
            score += 1.5

        entry = arr["close"][i] - p["slippage"]
        stop = arr["high"][i] + p["sw_stop_buffer"]
        risk = stop - entry
        if risk <= 0:
            return None
        target = entry - risk * p["sw_rr"]
        return ("sweep", -1, score, entry, stop, target, "sweep_short")

    if arr["low"][i] < recent_low - p["sw_sweep_buffer"] and arr["close"][i] > recent_low:
        score += 2.0
        if arr["body"][i] >= atr * p["sw_rejection_body_atr"]:
            score += 1.5
        if arr["close"][i] > arr["open"][i]:
            score += 1.5

        entry = arr["close"][i] + p["slippage"]
        stop = arr["low"][i] - p["sw_stop_buffer"]
        risk = entry - stop
        if risk <= 0:
            return None
        target = entry + risk * p["sw_rr"]
        return ("sweep", 1, score, entry, stop, target, "sweep_long")

    return None


def calc_smc_signal(arr: Dict[str, np.ndarray], i: int, p: Dict):
    if i < 25:
        return None

    hour = int(arr["hour"][i])
    if p["use_session"] and not in_session(hour, p["session_start"], p["session_end"]):
        return None

    prev = i - 1
    fast = arr[f"ema_{p['ema_fast']}"][prev]
    slow = arr[f"ema_{p['ema_slow']}"][prev]
    hh = arr[f"hh_{p['smc_bos_lookback']}"][prev]
    ll = arr[f"ll_{p['smc_bos_lookback']}"][prev]
    if np.isnan(hh) or np.isnan(ll):
        return None

    bullish_bias = fast > slow
    bearish_bias = fast < slow
    bullish_bos = arr["close"][i] > hh + p["smc_bos_buffer"]
    bearish_bos = arr["close"][i] < ll - p["smc_bos_buffer"]
    displacement = arr["body"][i] >= arr["atr_14"][i] * p["smc_displacement_body_atr"]
    near_ema = abs(arr["close"][i] - arr["ema_21"][prev]) <= arr["atr_14"][i] * p["smc_pullback_atr"]

    score = 0.0
    if bullish_bias:
        score += 1.5
    if bearish_bias:
        score += 1.5
    if displacement:
        score += 1.5
    if near_ema:
        score += 1.5

    if bullish_bias and bullish_bos and near_ema:
        entry = arr["close"][i] + p["slippage"]
        stop = arr["low"][i] - p["smc_stop_buffer"]
        risk = entry - stop
        if risk <= 0:
            return None
        target = entry + risk * p["smc_rr"]
        return ("smc", 1, score, entry, stop, target, "smc_long")

    if bearish_bias and bearish_bos and near_ema:
        entry = arr["close"][i] - p["slippage"]
        stop = arr["high"][i] + p["smc_stop_buffer"]
        risk = stop - entry
        if risk <= 0:
            return None
        target = entry - risk * p["smc_rr"]
        return ("smc", -1, score, entry, stop, target, "smc_short")

    return None


def choose_best_signal(signals, p: Dict):
    valid = [s for s in signals if s is not None]
    if not valid:
        return None

    filtered = []
    for s in valid:
        family = s[0]
        score = s[2]
        if score >= p[f"{family}_min_score"]:
            filtered.append(s)

    if not filtered:
        return None

    filtered.sort(key=lambda x: x[2], reverse=True)
    return filtered[0]


def manage_trade(arr: Dict[str, np.ndarray], signal, i: int, end_idx: int, p: Dict):
    family, direction, score, entry_price, stop_price, target_price, reason = signal
    max_hold = p["global_max_hold_bars"]
    last_idx = min(i + max_hold, end_idx - 1)

    exit_price = None
    exit_reason = None
    exit_idx = None

    for j in range(i + 1, last_idx + 1):
        high = arr["high"][j]
        low = arr["low"][j]

        if direction == 1:
            if low <= stop_price:
                exit_price = stop_price
                exit_reason = "stop"
                exit_idx = j
                break
            if high >= target_price:
                exit_price = target_price
                exit_reason = "target"
                exit_idx = j
                break
        else:
            if high >= stop_price:
                exit_price = stop_price
                exit_reason = "stop"
                exit_idx = j
                break
            if low <= target_price:
                exit_price = target_price
                exit_reason = "target"
                exit_idx = j
                break

    if exit_price is None:
        exit_idx = last_idx
        exit_price = arr["close"][last_idx]
        exit_reason = "timeout"

    if direction == 1:
        result_points = exit_price - entry_price
        risk = entry_price - stop_price
    else:
        result_points = entry_price - exit_price
        risk = stop_price - entry_price

    result_r = result_points / risk if risk > 0 else 0.0

    return {
        "family": family,
        "entry_idx": i,
        "exit_idx": exit_idx,
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "result_points": result_points,
        "result_r": result_r,
        "exit_reason": exit_reason,
        "reason": reason,
    }


def run_backtest_multi_setup(arr: Dict[str, np.ndarray], start_idx: int, end_idx: int, p: Dict) -> List[Dict]:
    trades: List[Dict] = []
    i = max(35, start_idx + 1)

    while i < end_idx - 2:
        old_i = i
        bo = calc_breakout_signal(arr, i, p)
        sw = calc_sweep_signal(arr, i, p)
        smc = calc_smc_signal(arr, i, p)

        best = choose_best_signal([bo, sw, smc], p)

        if best is None:
            i += 1
            continue

        trade = manage_trade(arr, best, i, end_idx, p)
        trades.append(trade)
        i = trade["exit_idx"] + 1

        if i == old_i:
            i += 1

    return trades


# =========================================================
# METRICS / REPORTS
# =========================================================
def calc_metrics(trades: List[Dict]) -> Dict:
    if not trades:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "timeouts": 0,
            "win_rate": 0.0,
            "total_points": 0.0,
            "total_r": 0.0,
            "avg_r": 0.0,
            "profit_factor": 0.0,
            "max_dd_r": 0.0,
            "avg_hold_bars": 0.0,
            "max_win_streak": 0,
            "max_loss_streak": 0,
        }

    arr_r = np.array([t["result_r"] for t in trades], dtype=float)
    arr_p = np.array([t["result_points"] for t in trades], dtype=float)
    holds = np.array([t["exit_idx"] - t["entry_idx"] for t in trades], dtype=float)

    wins = int(np.sum(arr_r > 0))
    losses = int(np.sum(arr_r < 0))
    timeouts = int(sum(1 for t in trades if t["exit_reason"] == "timeout"))

    gross_profit = float(np.sum(arr_r[arr_r > 0])) if np.any(arr_r > 0) else 0.0
    gross_loss = abs(float(np.sum(arr_r[arr_r < 0]))) if np.any(arr_r < 0) else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    equity = np.cumsum(arr_r)
    peaks = np.maximum.accumulate(equity) if len(equity) else np.array([0.0])
    dd = peaks - equity if len(equity) else np.array([0.0])
    max_dd_r = float(np.max(dd)) if len(dd) else 0.0

    max_win_streak = 0
    max_loss_streak = 0
    cur_w = 0
    cur_l = 0
    for x in arr_r:
        if x > 0:
            cur_w += 1
            cur_l = 0
        elif x < 0:
            cur_l += 1
            cur_w = 0
        else:
            cur_w = 0
            cur_l = 0
        max_win_streak = max(max_win_streak, cur_w)
        max_loss_streak = max(max_loss_streak, cur_l)

    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "win_rate": (wins / len(trades)) * 100.0,
        "total_points": float(np.sum(arr_p)),
        "total_r": float(np.sum(arr_r)),
        "avg_r": float(np.mean(arr_r)),
        "profit_factor": pf,
        "max_dd_r": max_dd_r,
        "avg_hold_bars": float(np.mean(holds)) if len(holds) else 0.0,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
    }


def build_trades_df(trades: List[Dict], arr: Dict[str, np.ndarray], strategy_name: str) -> pd.DataFrame:
    rows = []
    for t in trades:
        exit_idx = t["exit_idx"]
        rows.append({
            "strategy": strategy_name,
            "family": t["family"],
            "entry_idx": t["entry_idx"],
            "exit_idx": exit_idx,
            "entry_time": arr["time"][t["entry_idx"]],
            "exit_time": arr["time"][exit_idx],
            "month": arr["month"][exit_idx],
            "date": arr["date"][exit_idx],
            "direction": t["direction"],
            "entry_price": t["entry_price"],
            "exit_price": t["exit_price"],
            "result_points": t["result_points"],
            "result_r": t["result_r"],
            "exit_reason": t["exit_reason"],
            "reason": t["reason"],
        })
    return pd.DataFrame(rows)


def monthly_report(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()

    grp = trades_df.groupby(["strategy", "month"], as_index=False).agg(
        trades=("result_r", "count"),
        total_r=("result_r", "sum"),
        avg_r=("result_r", "mean"),
        win_rate=("result_r", lambda x: (x.gt(0).mean() * 100.0)),
    )
    return grp


def stress_test_slippage(arr: Dict[str, np.ndarray], params: Dict, strategy_name: str, extra_slippages=(0, 1, 2, 3)) -> pd.DataFrame:
    rows = []
    n = len(arr["close"])
    for extra in extra_slippages:
        p = dict(params)
        p["slippage"] = p["slippage"] + extra
        trades = run_backtest_multi_setup(arr, 0, n, p)
        m = calc_metrics(trades)
        rows.append({
            "strategy": strategy_name,
            "extra_slippage": extra,
            **m,
        })
    return pd.DataFrame(rows)


def monte_carlo(trades_df: pd.DataFrame, strategy_name: str, iterations: int = 1000, seed: int = 42) -> Dict:
    if trades_df.empty:
        return {
            "strategy": strategy_name,
            "mc_iterations": iterations,
            "mc_p5_total_r": 0.0,
            "mc_p50_total_r": 0.0,
            "mc_p95_total_r": 0.0,
            "mc_p95_max_dd_r": 0.0,
        }

    rng = np.random.default_rng(seed)
    arr_r = trades_df["result_r"].to_numpy(dtype=float)

    total_rs = []
    max_dds = []

    for _ in range(iterations):
        shuffled = rng.permutation(arr_r)
        equity = np.cumsum(shuffled)
        peaks = np.maximum.accumulate(equity)
        dd = peaks - equity
        total_rs.append(float(np.sum(shuffled)))
        max_dds.append(float(np.max(dd)) if len(dd) else 0.0)

    return {
        "strategy": strategy_name,
        "mc_iterations": iterations,
        "mc_p5_total_r": float(np.percentile(total_rs, 5)),
        "mc_p50_total_r": float(np.percentile(total_rs, 50)),
        "mc_p95_total_r": float(np.percentile(total_rs, 95)),
        "mc_p95_max_dd_r": float(np.percentile(max_dds, 95)),
    }


def walkforward_report(arr: Dict[str, np.ndarray], params: Dict, strategy_name: str, windows: int = 5) -> pd.DataFrame:
    n = len(arr["close"])
    rows = []
    if n < 500:
        return pd.DataFrame()

    step = max(1, n // (windows + 2))
    train = step * 2
    valid = step
    test = step

    start = 0
    wf_id = 0
    while start + train + valid + test <= n:
        wf_id += 1
        a = start
        b = a + train
        c = b + valid
        d = c + test

        tr = run_backtest_multi_setup(arr, a, b, params)
        va = run_backtest_multi_setup(arr, b, c, params)
        te = run_backtest_multi_setup(arr, c, d, params)

        m_tr = calc_metrics(tr)
        m_va = calc_metrics(va)
        m_te = calc_metrics(te)

        rows.append({
            "strategy": strategy_name,
            "wf_id": wf_id,
            "train_trades": m_tr["trades"],
            "train_pf": m_tr["profit_factor"],
            "train_total_r": m_tr["total_r"],
            "valid_trades": m_va["trades"],
            "valid_pf": m_va["profit_factor"],
            "valid_total_r": m_va["total_r"],
            "test_trades": m_te["trades"],
            "test_pf": m_te["profit_factor"],
            "test_total_r": m_te["total_r"],
            "test_max_dd_r": m_te["max_dd_r"],
        })

        start += step

    return pd.DataFrame(rows)


def portfolio_report(trades_all: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    if trades_all.empty:
        return pd.DataFrame(), {}

    pf = trades_all.groupby("date", as_index=False).agg(
        portfolio_r=("result_r", "sum"),
        trades=("result_r", "count"),
    )
    pf["equity"] = pf["portfolio_r"].cumsum()
    pf["peak"] = pf["equity"].cummax()
    pf["dd"] = pf["peak"] - pf["equity"]

    arr_r = pf["portfolio_r"].to_numpy(dtype=float)
    gross_profit = float(np.sum(arr_r[arr_r > 0])) if np.any(arr_r > 0) else 0.0
    gross_loss = abs(float(np.sum(arr_r[arr_r < 0]))) if np.any(arr_r < 0) else 0.0
    pfactor = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    summary = {
        "portfolio_days": int(len(pf)),
        "portfolio_total_r": float(np.sum(arr_r)),
        "portfolio_avg_r_day": float(np.mean(arr_r)) if len(arr_r) else 0.0,
        "portfolio_pf": pfactor,
        "portfolio_max_dd_r": float(pf["dd"].max()) if len(pf) else 0.0,
    }
    return pf, summary


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Institutional validation for top 5 strategies")
    parser.add_argument("--csv", type=str, default="wdo_m5.csv")
    parser.add_argument("--mc-iterations", type=int, default=1000)
    parser.add_argument("--outdir", type=str, default="institutional_reports")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_csv(args.csv)
    df = add_features(df)
    arr = build_arrays(df)
    n = len(df)

    summary_rows = []
    monthly_frames = []
    stress_frames = []
    walk_frames = []
    mc_rows = []
    all_trades_frames = []

    for cfg in TOP5_CONFIGS:
        name = cfg["name"]
        params = cfg["params"]

        trades = run_backtest_multi_setup(arr, 0, n, params)
        metrics = calc_metrics(trades)
        trades_df = build_trades_df(trades, arr, name)

        summary_rows.append({
            "strategy": name,
            **metrics,
        })

        trades_df.to_csv(os.path.join(args.outdir, f"{name}_trades.csv"), index=False)
        all_trades_frames.append(trades_df)

        mon = monthly_report(trades_df)
        if not mon.empty:
            monthly_frames.append(mon)

        stress = stress_test_slippage(arr, params, name, extra_slippages=(0, 1, 2, 3))
        stress_frames.append(stress)

        wf = walkforward_report(arr, params, name, windows=5)
        if not wf.empty:
            walk_frames.append(wf)

        mc = monte_carlo(trades_df, name, iterations=args.mc_iterations, seed=42)
        mc_rows.append(mc)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["profit_factor", "total_r", "win_rate"],
        ascending=[False, False, False],
    )
    summary_df.to_csv(os.path.join(args.outdir, "top5_individual_report.csv"), index=False)

    if monthly_frames:
        monthly_df = pd.concat(monthly_frames, ignore_index=True)
        monthly_df.to_csv(os.path.join(args.outdir, "top5_monthly_report.csv"), index=False)

    if stress_frames:
        stress_df = pd.concat(stress_frames, ignore_index=True)
        stress_df.to_csv(os.path.join(args.outdir, "top5_stress_report.csv"), index=False)

    if walk_frames:
        walk_df = pd.concat(walk_frames, ignore_index=True)
        walk_df.to_csv(os.path.join(args.outdir, "top5_walkforward_report.csv"), index=False)

    mc_df = pd.DataFrame(mc_rows)
    mc_df.to_csv(os.path.join(args.outdir, "top5_montecarlo_report.csv"), index=False)

    if all_trades_frames:
        all_trades = pd.concat(all_trades_frames, ignore_index=True)
        all_trades.to_csv(os.path.join(args.outdir, "top5_all_trades.csv"), index=False)

        pf_df, pf_summary = portfolio_report(all_trades)
        if not pf_df.empty:
            pf_df.to_csv(os.path.join(args.outdir, "top5_portfolio_daily.csv"), index=False)

        pd.DataFrame([pf_summary]).to_csv(
            os.path.join(args.outdir, "top5_portfolio_report.csv"), index=False
        )

    with open(os.path.join(args.outdir, "top5_configs_used.json"), "w", encoding="utf-8") as f:
        json.dump(TOP5_CONFIGS, f, indent=2, ensure_ascii=False)

    print("Relatórios salvos em:", args.outdir)
    print("- top5_individual_report.csv")
    print("- top5_monthly_report.csv")
    print("- top5_stress_report.csv")
    print("- top5_walkforward_report.csv")
    print("- top5_montecarlo_report.csv")
    print("- top5_portfolio_report.csv")
    print("- top5_all_trades.csv")


if __name__ == "__main__":
    main()