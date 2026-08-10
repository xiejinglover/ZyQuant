from __future__ import annotations

from datetime import date

import pandas as pd


CODE_A = "510300.XSHG"
CODE_B = "600000.XSHG"


def canonical_tables():
    days = [item.date() for item in pd.bdate_range("2025-01-02", periods=9)]
    instruments = pd.DataFrame([
        {
            "instrument_id": CODE_A, "symbol": "510300", "exchange": "XSHG",
            "asset_type": "etf", "list_date": date(2012, 1, 1),
            "delist_date": None, "lot_size": 100, "sell_delay_days": 0,
        },
        {
            "instrument_id": CODE_B, "symbol": "600000", "exchange": "XSHG",
            "asset_type": "stock", "list_date": date(1999, 11, 10),
            "delist_date": None, "lot_size": 100, "sell_delay_days": 1,
        },
    ])
    calendar = pd.DataFrame({"trade_date": days, "exchange": "XSHG"})
    closes_a = [10.0, 10.2, 10.5, 9.5, 9.7, 9.8, 10.0, 10.1, 10.2]
    closes_b = [8.0, 8.1, 8.0, 8.2, 8.3, 8.4, 8.5, 8.4, 8.6]
    raw_rows = []
    for code, values in ((CODE_A, closes_a), (CODE_B, closes_b)):
        for index, (day, close) in enumerate(zip(days, values, strict=True)):
            previous = values[index - 1] if index else close
            raw_rows.append({
                "trade_date": day, "instrument_id": code,
                "open": close, "high": close * 1.01, "low": close * 0.99,
                "close": close, "pre_close": previous,
                "volume": 10_000_000, "amount": 100_000_000.0,
                "paused": False, "limit_up": close * 1.1, "limit_down": close * 0.9,
                "source_updated_at": day, "source_batch_id": "sample",
            })
    raw = pd.DataFrame(raw_rows)
    actions = pd.DataFrame([
        {
            "event_id": "div-a", "instrument_id": CODE_A,
            "event_type": "cash_dividend", "record_date": days[2],
            "ex_date": days[3], "pay_date": days[4], "cash_per_share": 1.0,
            "share_ratio": 0.0, "status": "active", "announced_at": days[1],
        }
    ])
    membership = pd.DataFrame([
        {
            "universe_id": "TEST", "instrument_id": code,
            "effective_from": days[0], "effective_to": None, "known_at": days[0],
        } for code in (CODE_A, CODE_B)
    ])
    industry = pd.DataFrame([
        {
            "classification": "TEST", "industry_id": group,
            "instrument_id": code, "effective_from": days[0],
            "effective_to": None, "known_at": days[0],
        } for code, group in ((CODE_A, "ETF"), (CODE_B, "BANK"))
    ])
    market_rules = pd.DataFrame([
        {
            "rule_id": f"XSHG-{asset_type}-v1", "exchange": "XSHG",
            "asset_type": asset_type, "effective_from": days[0],
            "effective_to": None, "commission_bps": 2.5,
            "minimum_commission": 5.0,
            "sell_tax_bps": 5.0 if asset_type == "stock" else 0.0,
            "buy_tax_bps": 0.0, "transfer_fee_bps": 0.0, "currency": "CNY",
        }
        for asset_type in ("stock", "etf")
    ])
    return {
        "instruments": instruments,
        "trade_calendar": calendar,
        "daily_raw": raw,
        "corporate_actions": actions,
        "universe_membership": membership,
        "industry_membership": industry,
        "market_rules": market_rules,
    }, days


def signal_frame(days):
    rows = []
    for index, day in enumerate(days):
        winner = CODE_A if index < 4 else CODE_B
        for code in (CODE_A, CODE_B):
            rows.append({
                "signal_date": day, "instrument_id": code,
                "score": 1.0 if code == winner else 0.0,
            })
    return pd.DataFrame(rows)
