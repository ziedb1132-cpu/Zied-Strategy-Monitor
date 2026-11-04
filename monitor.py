#!/usr/bin/env python3
"""Live paper monitor for one validated Zied Strategy Lab BUY strategy."""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from strategy_lab import _clean, extract_features

ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "monitor_state.json"
MEMORY_URL = (
    "https://raw.githubusercontent.com/ziedb1132-cpu/"
    "Zied-Strategy-Lab/main/data/lab_memory.json"
)
SUMMARY_URL = (
    "https://raw.githubusercontent.com/ziedb1132-cpu/"
    "Zied-Strategy-Lab/main/data/dashboard_summary.json"
)
OKX_URL = "https://www.okx.com/api/v5"
SUPPORTED_BARS = {"5m", "15m", "30m", "1H", "4H", "1D"}
RUN_SECONDS = 5 * 60 * 60
POLL_SECONDS = 10
PUBLISH_SECONDS = 60


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime | None = None) -> str:
    return (value or utc_now()).isoformat(timespec="seconds")


def http_json(url: str, params: dict[str, str] | None = None) -> dict[str, Any]:
    if params:
        url += "?" + urlencode(params)
    request = Request(url, headers={
        "User-Agent": "Zied-Strategy-Monitor/1.0",
        "Accept": "application/json",
    })
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if str(payload.get("code", "0")) != "0":
                raise RuntimeError(f"API error {payload.get('code')}: {payload.get('msg')}")
            return payload
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f"Request failed: {last_error}") from last_error


def resolve_strategy_choice(choice: str) -> dict[str, Any]:
    """Resolve a live rank slot against the latest HTML-equivalent top-20 list."""
    value = (choice or "").strip()
    rank_match = re.fullmatch(r"TOP[_-]?(\d{1,2})", value, re.I)
    if rank_match:
        rank = int(rank_match.group(1))
        if not 1 <= rank <= 20:
            raise ValueError("Strategy rank must be between TOP_1 and TOP_20.")
        summary = http_json(SUMMARY_URL)
        strategies = [
            item for item in (summary.get("strategies") or [])
            if isinstance(item, dict) and item.get("id")
        ]
        strategies.sort(
            key=lambda item: (
                float(item.get("rating", 0.0)),
                float(item.get("test_net_total_return", 0.0)),
                float(item.get("test_profit_factor", 0.0)),
            ),
            reverse=True,
        )
        if len(strategies) < rank:
            raise ValueError(
                f"Strategy Lab currently publishes only {len(strategies)} qualified "
                f"strategies in its live top-20 list; TOP_{rank} is not available yet."
            )
        selected = strategies[rank - 1]
        return {
            "id": str(selected["id"]).lower(),
            "name": str(selected.get("name") or selected["id"]),
            "rating": float(selected.get("rating", 0.0)),
            "rank": rank,
            "ranking_cycle": int(summary.get("cycle", 0)),
            "ranking_updated_at": str(summary.get("updated_at") or ""),
        }

    match = re.match(r"^([0-9a-f]{12})", value, re.I)
    if not match:
        raise ValueError("Choose a live rank from TOP_1 to TOP_20.")
    return {"id": match.group(1).lower(), "name": value, "rating": None,
            "rank": None, "ranking_cycle": None, "ranking_updated_at": None}


def load_candidate(strategy_id: str, requested_bar: str) -> dict[str, Any]:
    payload = http_json(MEMORY_URL)
    stats = payload.get("performance", {}).get(strategy_id)
    if not stats:
        raise ValueError(
            f"Strategy {strategy_id} is not present in the latest Strategy Lab memory. "
            "Refresh the workflow strategy choices from the current dashboard summary."
        )
    if not stats.get("successful", False):
        raise ValueError("The selected strategy is no longer marked successful by Strategy Lab.")
    candidate = stats.get("candidate") or {}
    if candidate.get("direction") != "long":
        raise ValueError("Only BUY / long strategies can be monitored.")
    actual_bar = str(candidate.get("bar", ""))
    if requested_bar != "AUTO" and actual_bar != requested_bar:
        raise ValueError(
            f"Timeframe mismatch: this strategy belongs to {actual_bar}, "
            f"but the workflow was started with {requested_bar}. Select AUTO or the matching timeframe."
        )
    if not candidate.get("features"):
        raise ValueError("The selected strategy has no saved feature definition.")
    return {
        "id": strategy_id,
        "bar": actual_bar,
        "candidate": candidate,
        "holding_bars": int(candidate.get("holding_bars", 0)),
        "threshold": float(candidate.get("threshold", 0.0)),
    }


def fetch_market(inst_id: str, bar: str) -> tuple[pd.DataFrame, float]:
    ticker_payload = http_json(f"{OKX_URL}/market/ticker", {"instId": inst_id})
    tickers = ticker_payload.get("data") or []
    if not tickers:
        raise RuntimeError(f"OKX returned no ticker for {inst_id}.")
    current_price = float(tickers[0]["last"])
    if not math.isfinite(current_price) or current_price <= 0:
        raise RuntimeError("OKX returned an invalid last price.")

    candles_payload = http_json(
        f"{OKX_URL}/market/candles",
        {"instId": inst_id, "bar": bar, "limit": "300"},
    )
    raw_rows = candles_payload.get("data") or []
    rows = []
    for row in reversed(raw_rows):
        if len(row) < 6:
            continue
        try:
            rows.append({
                "timestamp": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            })
        except (TypeError, ValueError):
            continue
    if len(rows) < 150:
        raise RuntimeError(f"Only {len(rows)} candles available; at least 150 are needed.")
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    return frame, current_price


def compute_strategy(frame: pd.DataFrame, candidate: dict[str, Any]) -> dict[str, Any]:
    clean = _clean(frame)
    features = extract_features(frame)
    columns = [name for name in candidate["features"] if name in features.columns]
    if not columns:
        raise RuntimeError("None of the selected strategy's features are available.")
    values = features[columns].astype(float)
    means = values.rolling(100, min_periods=30).mean()
    std = values.rolling(100, min_periods=30).std()
    strength = (
        ((values - means) / (std + 1e-9))
        .clip(-4, 4)
        .fillna(0)
        .mean(axis=1)
    )
    score = float(strength.iloc[-1])
    close = float(clean.iloc[-1]["close"])
    atr_ratio = float(features["atr_14_raw"].iloc[-1]) if "atr_14_raw" in features else 0.0
    atr = atr_ratio * close if atr_ratio > 0 else 0.0
    candle_ms = int(pd.Timestamp(clean.iloc[-1]["timestamp"]).timestamp() * 1000)
    return {
        "buy_signal": score > float(candidate.get("threshold", 0.0)),
        "strength": score,
        "threshold": float(candidate.get("threshold", 0.0)),
        "entry_reference": close,
        "atr": atr,
        "candle_ms": candle_ms,
        "clean": clean,
    }


def read_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        try:
            value = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
        except (OSError, json.JSONDecodeError):
            pass
    return {"schema_version": 1, "status": "idle", "running": False}


def save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = iso()
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git_publish() -> None:
    subprocess.run(["git", "add", "monitor_state.json"], cwd=ROOT, check=True)
    changed = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", "monitor_state.json"],
        cwd=ROOT,
    )
    if changed.returncode == 0:
        return
    if changed.returncode != 1:
        raise RuntimeError("Unable to inspect staged monitor state.")
    subprocess.run(
        ["git", "commit", "-m", "Update live strategy monitor state [skip ci]"],
        cwd=ROOT, check=True,
    )
    for attempt in range(1, 6):
        subprocess.run(["git", "fetch", "origin", "main"], cwd=ROOT, check=True)
        rebase = subprocess.run(
            ["git", "rebase", "origin/main"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if rebase.returncode != 0:
            subprocess.run(["git", "rebase", "--abort"], cwd=ROOT, check=False)
            raise RuntimeError("Could not safely rebase monitor state: " + rebase.stdout[-1500:])
        push = subprocess.run(
            ["git", "push", "origin", "HEAD:main"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if push.returncode == 0:
            print("Published monitor_state.json to main.", flush=True)
            return
        print(f"Push attempt {attempt}/5 failed: {push.stdout[-800:]}", flush=True)
        time.sleep(attempt)
    raise RuntimeError("Could not publish monitor_state.json after five attempts.")


def start_or_resume_state(old: dict[str, Any], strategy: dict[str, Any],
                          strategy_name: str, inst_id: str, bar: str) -> dict[str, Any]:
    same_session = (
        old.get("strategy_id") == strategy["id"]
        and old.get("symbol") == inst_id
        and old.get("timeframe") == bar
    )
    now = utc_now()
    if not same_session:
        old = {
            "schema_version": 1,
            "strategy_id": strategy["id"],
            "strategy_name": strategy_name,
            "symbol": inst_id,
            "timeframe": bar,
            "started_at": iso(now),
            "buy_now": None,
            "buy_at": None,
            "sl": None,
            "tp": None,
            "active_signal": False,
            "signal_status": "WAITING",
            "last_signal_candle_ms": None,
            "entry_candle_ms": None,
            "last_exit_reason": None,
            "last_exit_at": None,
            "cycle_count": 0,
        }
    old["strategy_name"] = strategy_name
    old["strategy_id"] = strategy["id"]
    old["symbol"] = inst_id
    old["timeframe"] = bar
    old["status"] = "running"
    old["running"] = True
    old["batch_started_at"] = iso(now)
    old["ends_at"] = iso(now + timedelta(seconds=RUN_SECONDS))
    old["holding_bars"] = strategy["holding_bars"]
    old["threshold"] = strategy["threshold"]
    old["note"] = "Paper monitoring is active. Waiting for a BUY signal or tracking the current signal."
    old["cycle_count"] = int(old.get("cycle_count", 0))
    return old


def process_tick(state: dict[str, Any], frame: pd.DataFrame, price: float,
                 candidate: dict[str, Any], strategy: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    result = compute_strategy(frame, candidate)
    now = utc_now()
    state["buy_now"] = price
    state["price_updated_at"] = iso(now)
    state["strategy_strength"] = result["strength"]
    state["strategy_threshold"] = result["threshold"]
    state["last_completed_candle_at"] = iso(
        pd.Timestamp(result["candle_ms"], unit="ms", tz="UTC").to_pydatetime()
    )
    state["cycle_count"] = int(state.get("cycle_count", 0)) + 1
    event = False

    if state.get("active_signal"):
        entry_ms = state.get("entry_candle_ms")
        clean_times = [
            int(pd.Timestamp(v).timestamp() * 1000)
            for v in frame["timestamp"].tolist()
        ]
        try:
            entry_index = clean_times.index(int(entry_ms))
            latest_index = len(clean_times) - 2
            age_bars = max(0, latest_index - entry_index)
        except (ValueError, TypeError):
            age_bars = int(state.get("holding_bars", 0)) + 1
        state["signal_age_bars"] = age_bars
        sl = float(state["sl"])
        tp = float(state["tp"])
        exit_reason = None
        if price <= sl:
            exit_reason = "SL"
        elif price >= tp:
            exit_reason = "TP"
        elif age_bars >= int(strategy["holding_bars"]):
            exit_reason = "TIMEOUT"
        if exit_reason:
            entry_price = float(state.get("buy_at") or 0.0)
            exit_price = float(price)
            gross_pct = ((exit_price / entry_price) - 1.0) * 100.0 if entry_price > 0 else 0.0
            # Approximate round-trip spot fees: 0.1% entry + 0.1% exit.
            net_pct = gross_pct - 0.20
            state["trade_count"] = int(state.get("trade_count", 0)) + 1
            state["realized_net_return_pct"] = float(state.get("realized_net_return_pct", 0.0)) + net_pct
            state["closed_trades"] = list(state.get("closed_trades", []))
            state["closed_trades"].append({
                "opened_at": state.get("signal_opened_at"),
                "closed_at": iso(now),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "gross_return_pct": round(gross_pct, 6),
                "net_return_pct": round(net_pct, 6),
            })
            state["closed_trades"] = state["closed_trades"][-500:]
            if exit_reason == "TP":
                state["tp_count"] = int(state.get("tp_count", 0)) + 1
            elif exit_reason == "SL":
                state["sl_count"] = int(state.get("sl_count", 0)) + 1
            else:
                state["timeout_count"] = int(state.get("timeout_count", 0)) + 1
            if net_pct > 0:
                state["winning_trades"] = int(state.get("winning_trades", 0)) + 1
                state["gross_profit_pct"] = float(state.get("gross_profit_pct", 0.0)) + net_pct
            elif net_pct < 0:
                state["losing_trades"] = int(state.get("losing_trades", 0)) + 1
                state["gross_loss_pct"] = float(state.get("gross_loss_pct", 0.0)) + abs(net_pct)
            gross_profit = float(state.get("gross_profit_pct", 0.0))
            gross_loss = float(state.get("gross_loss_pct", 0.0))
            state["profit_factor"] = round(gross_profit / gross_loss, 4) if gross_loss > 0 else (None if gross_profit == 0 else "∞")
            state["last_exit_price"] = exit_price
            state["last_trade_net_return_pct"] = round(net_pct, 4)
            state["active_signal"] = False
            state["signal_status"] = exit_reason
            state["last_exit_reason"] = exit_reason
            state["last_exit_at"] = iso(now)
            state["note"] = f"Last paper signal closed by {exit_reason} · net {net_pct:+.2f}% after estimated fees."
            event = True
        else:
            state["signal_status"] = "ACTIVE"
            state["note"] = (
                f"BUY signal active · age {age_bars}/{strategy['holding_bars']} bars · "
                "paper tracking only."
            )

    if result["buy_signal"]:
        candle_ms = result["candle_ms"]
        is_new_signal_candle = candle_ms != state.get("last_signal_candle_ms")
        if is_new_signal_candle:
            state["last_signal_candle_ms"] = candle_ms
        if not state.get("active_signal") and is_new_signal_candle:
            if result["atr"] > 0 and result["entry_reference"] > 0:
                entry = float(result["entry_reference"])
                atr = float(result["atr"])
                state["buy_at"] = entry
                state["sl"] = entry - 2.0 * atr
                state["tp"] = entry + 3.0 * atr
                state["active_signal"] = True
                state["signal_status"] = "ACTIVE"
                state["entry_candle_ms"] = candle_ms
                state["signal_age_bars"] = 0
                state["signal_opened_at"] = iso(now)
                state["note"] = (
                    f"New BUY signal detected · entry {entry:.10g} · "
                    "SL = entry - 2xATR(14), TP = entry + 3xATR(14)."
                )
                event = True

    if not state.get("active_signal") and state.get("signal_status") in (None, "WAITING"):
        state["note"] = (
            f"Monitoring {state['strategy_name']} · waiting for BUY "
            f"(strength {result['strength']:.3f}, threshold {result['threshold']:.3f})."
        )
    return state, event


def run_monitor(strategy_choice: str, bar: str, inst_id: str) -> None:
    selected = resolve_strategy_choice(strategy_choice)
    strategy_id = selected["id"]
    if bar != "AUTO" and bar not in SUPPORTED_BARS:
        raise ValueError(f"Unsupported timeframe {bar}.")
    inst_id = inst_id.strip().upper()
    if not re.fullmatch(r"[A-Z0-9]+-USDT", inst_id):
        raise ValueError("Instrument must be an OKX Spot USDT pair, e.g. SOL-USDT.")
    old_state = read_state()
    # Never abandon an open paper signal merely because the live ranking changed.
    if old_state.get("active_signal") and old_state.get("strategy_id"):
        active_id = str(old_state["strategy_id"]).lower()
        active_strategy = load_candidate(active_id, "AUTO")
        selected = {
            "id": active_id,
            "name": str(old_state.get("strategy_name") or active_id),
            "rating": old_state.get("strategy_rating"),
            "rank": old_state.get("strategy_rank"),
            "ranking_cycle": old_state.get("ranking_cycle"),
            "ranking_updated_at": old_state.get("ranking_updated_at"),
        }
        strategy = active_strategy
    else:
        strategy = load_candidate(strategy_id, bar)
    bar = strategy["bar"]
    strategy_name = selected["name"]
    state = start_or_resume_state(old_state, strategy, strategy_name, inst_id, bar)
    state["strategy_rating"] = selected.get("rating")
    state.setdefault("trade_count", 0)
    state.setdefault("tp_count", 0)
    state.setdefault("sl_count", 0)
    state.setdefault("timeout_count", 0)
    state.setdefault("winning_trades", 0)
    state.setdefault("losing_trades", 0)
    state.setdefault("gross_profit_pct", 0.0)
    state.setdefault("gross_loss_pct", 0.0)
    state.setdefault("profit_factor", None)
    state.setdefault("realized_net_return_pct", 0.0)
    state["strategy_rank"] = selected.get("rank")
    state["ranking_cycle"] = selected.get("ranking_cycle")
    state["ranking_updated_at"] = selected.get("ranking_updated_at")
    # Fetch and publish a market price before the first page render.
    try:
        _, initial_price = fetch_market(inst_id, bar)
        state["buy_now"] = initial_price
        state["price_updated_at"] = iso()
        state.pop("last_error", None)
    except Exception as exc:
        state["last_error"] = str(exc)[:500]
        state["note"] = f"Waiting for first market-price update: {str(exc)[:250]}"
    save_state(state)
    git_publish()

    next_publish = time.monotonic() + PUBLISH_SECONDS
    stop_at = time.monotonic() + RUN_SECONDS
    print(
        f"Monitoring {strategy_name} ({strategy_id}) on {inst_id} {bar} "
        "for five hours. Price poll: 10s; JSON publication: 60s.",
        flush=True,
    )
    while time.monotonic() < stop_at:
        try:
            frame, price = fetch_market(inst_id, bar)
            state, event = process_tick(state, frame, price, strategy["candidate"], strategy)
            state["running"] = True
            state["status"] = "running"
            state["note"] = state.get("note") or "Paper monitoring is active."
            save_state(state)
            if event or time.monotonic() >= next_publish:
                git_publish()
                next_publish = time.monotonic() + PUBLISH_SECONDS
        except Exception as exc:
            state["last_error"] = str(exc)[:500]
            state["note"] = f"Temporary monitoring error: {str(exc)[:300]}"
            save_state(state)
            if time.monotonic() >= next_publish:
                git_publish()
                next_publish = time.monotonic() + PUBLISH_SECONDS
            print(f"Monitor tick error: {exc}", flush=True)
        time.sleep(POLL_SECONDS)

    state["running"] = True
    state["status"] = "running"
    state["note"] = "Five-hour batch completed; the next workflow will resume this same strategy and state."
    save_state(state)
    git_publish()
    print("Five-hour batch complete. State saved for the next workflow.", flush=True)


def stop_monitor() -> None:
    state = read_state()
    state["running"] = False
    state["status"] = "stopped"
    state["stopped_at"] = iso()
    state["note"] = "Monitoring stopped manually from GitHub Actions. Saved signal levels are preserved."
    save_state(state)
    git_publish()
    print("Monitor state marked as stopped.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Five-hour OKX paper strategy monitor.")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--strategy-choice", default="")
    parser.add_argument("--bar", default="4H")
    parser.add_argument("--inst-id", default="SOL-USDT")
    args = parser.parse_args()
    if args.stop:
        stop_monitor()
    elif args.run:
        run_monitor(args.strategy_choice, args.bar, args.inst_id)
    else:
        parser.error("Choose --run or --stop.")


if __name__ == "__main__":
    main()
