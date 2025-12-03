from __future__ import annotations

import asyncio
import logging
from typing import Iterable, Sequence

import aiohttp
import numpy as np
import pandas as pd
from urllib.parse import urlencode

log = logging.getLogger(__name__)

# Hard-code the Odds API key; replace with your key if you want live odds ingestion.
# NOTE: The odds key is intentionally hard-coded for this pipeline run.
# Replace it if you need to use your own The Odds API credential.
ODDS_API_KEY = "5b6f0290e265c3329b3ed27897d79eaf"
ODDS_BASE = "https://api.the-odds-api.com/v4"
NFL_SPORT_KEY = "americanfootball_nfl"
ODDS_FORMAT = "american"

PROP_MARKETS = ",".join([
    "player_pass_yards",
    "player_pass_tds",
    "player_rush_yards",
    "player_rush_tds",
    "player_receptions",
    "player_receiving_yards",
    "player_receiving_tds",
    "player_anytime_td",
])

NFL_MARKET_MAP = {
    "h2h": "MONEYLINE",
    "spreads": "SPREAD",
    "totals": "TOTAL",
    "player_pass_yards": "PASS_YDS",
    "player_pass_tds": "PASS_TD",
    "player_rush_yards": "RUSH_YDS",
    "player_rush_tds": "RUSH_TD",
    "player_receptions": "RECEPTIONS",
    "player_receiving_yards": "REC_YDS",
    "player_receiving_tds": "REC_TD",
    "player_anytime_td": "ANY_TD",
    # Alternate keys occasionally returned by the API
    "player_pass_yds": "PASS_YDS",
    "player_rush_yds": "RUSH_YDS",
    "player_reception_yds": "REC_YDS",
}


def american_to_decimal(american: float) -> float:
    american = float(american)
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def american_to_prob(american: float) -> float:
    return 1.0 / american_to_decimal(american)


def _build_url(base: str, path: str, params: dict) -> str:
    base = base.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return f"{base}{path}?{urlencode(params, doseq=True)}"


async def _get_json(session: aiohttp.ClientSession, path: str, **params):
    params = {"apiKey": ODDS_API_KEY, **params}
    url = _build_url(ODDS_BASE, path, params)
    try:
        async with session.get(url, timeout=30) as r:
            if r.status != 200:
                txt = await r.text()
                log.warning("Odds GET %s -> %s\n%s", r.status, url, txt[:600])
                return None
            return await r.json()
    except Exception:
        log.exception("GET failed: %s", url)
        return None


async def fetch_nfl_game_odds(
    session: aiohttp.ClientSession,
    *,
    regions: str | Sequence[str] = ("us", "us2"),
    markets: str | Sequence[str] = ("h2h", "spreads", "totals"),
    odds_format: str = ODDS_FORMAT,
    date_format: str = "iso",
) -> pd.DataFrame:
    if not ODDS_API_KEY:
        log.warning("ODDS_API_KEY is not set; skipping odds fetch and returning empty frame.")
        return pd.DataFrame(
            columns=[
                "event_id",
                "commence_time",
                "home_team",
                "away_team",
                "book",
                "market",
                "side",
                "line",
                "american_odds",
                "decimal_odds",
                "imp_prob",
            ]
        )

    if isinstance(regions, (list, tuple)):
        regions = ",".join(regions)
    if isinstance(markets, (list, tuple)):
        markets = ",".join(markets)

    data = await _get_json(
        session,
        f"/sports/{NFL_SPORT_KEY}/odds",
        regions=regions,
        markets=markets,
        oddsFormat=odds_format,
        dateFormat=date_format,
    )
    if not data:
        return pd.DataFrame(
            columns=[
                "event_id",
                "commence_time",
                "home_team",
                "away_team",
                "book",
                "market",
                "side",
                "line",
                "american_odds",
                "decimal_odds",
                "imp_prob",
            ]
        )

    rows: list[dict] = []
    for ev in data:
        eid = str(ev.get("id", ""))
        home = ev.get("home_team") or ev.get("homeTeam") or ""
        away = ev.get("away_team") or ev.get("awayTeam") or ""
        kft = ev.get("commence_time") or ev.get("commenceTime") or ""

        for bm in (ev.get("bookmakers") or []):
            book = bm.get("key") or bm.get("title")
            for m in (bm.get("markets") or []):
                mkey = str(m.get("key", "")).lower()
                market_label = NFL_MARKET_MAP.get(mkey, mkey.upper())
                for o in (m.get("outcomes") or []):
                    name = o.get("name")
                    price = o.get("price")
                    line = o.get("point") or o.get("total") or o.get("handicap")

                    rows.append(
                        {
                            "event_id": eid,
                            "home_team": home,
                            "away_team": away,
                            "commence_time": kft,
                            "book": book,
                            "market": market_label,
                            "side": name,
                            "line": float(line) if line is not None else np.nan,
                            "american_odds": float(price) if price is not None else np.nan,
                        }
                    )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["decimal_odds"] = df["american_odds"].apply(american_to_decimal)
    df["imp_prob"] = df["american_odds"].apply(american_to_prob)
    if "line" in df.columns:
        df["line"] = pd.to_numeric(df["line"], errors="coerce")
    return df.reset_index(drop=True)


async def fetch_nfl_prop_odds(
    session: aiohttp.ClientSession,
    event_ids: str | Iterable[str],
    *,
    regions: str | Sequence[str] = ("us", "us2"),
    odds_format: str = ODDS_FORMAT,
    date_format: str = "iso",
) -> pd.DataFrame:
    if isinstance(event_ids, str):
        event_ids = [event_ids]
    event_ids = [e for e in (event_ids or []) if e]
    if not event_ids:
        return pd.DataFrame(
            columns=[
                "event_id",
                "commence_time",
                "home_team",
                "away_team",
                "book",
                "market",
                "player",
                "side",
                "line",
                "american_odds",
                "decimal_odds",
                "imp_prob",
            ]
        )

    if isinstance(regions, (list, tuple)):
        regions = ",".join(regions)

    tasks = [
        _get_json(
            session,
            f"/sports/{NFL_SPORT_KEY}/events/{eid}/odds",
            regions=regions,
            markets=PROP_MARKETS,
            oddsFormat=odds_format,
            dateFormat=date_format,
        )
        for eid in event_ids
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    rows: list[dict] = []
    for eid, data in zip(event_ids, results):
        if not data:
            continue
        home = data.get("home_team")
        away = data.get("away_team")
        kft = data.get("commence_time")

        for bm in (data.get("bookmakers") or []):
            book = bm.get("key") or bm.get("title")
            for m in (bm.get("markets") or []):
                mkey = str(m.get("key", "")).lower()
                market_label = NFL_MARKET_MAP.get(mkey, mkey.upper())
                for o in (m.get("outcomes") or []):
                    player = o.get("description") or o.get("participant") or ""
                    side = (o.get("name") or "").title()
                    if mkey == "player_anytime_td":
                        side = {"Yes": "Over", "No": "Under"}.get(side, side)
                    price = o.get("price")
                    line = o.get("point")
                    if not player or price is None:
                        continue
                    rows.append(
                        {
                            "event_id": eid,
                            "home_team": home,
                            "away_team": away,
                            "commence_time": kft,
                            "book": book,
                            "market": market_label,
                            "player": str(player),
                            "side": side,
                            "line": float(line) if line is not None else np.nan,
                            "american_odds": float(price),
                        }
                    )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["decimal_odds"] = df["american_odds"].apply(american_to_decimal)
    df["imp_prob"] = df["american_odds"].apply(american_to_prob)
    if "line" in df.columns:
        df["line"] = pd.to_numeric(df["line"], errors="coerce")
    return df.reset_index(drop=True)


async def event_ids_for_date(session: aiohttp.ClientSession, day_iso: str) -> list[str]:
    odds = await _get_json(
        session,
        f"/sports/{NFL_SPORT_KEY}/odds",
        regions="us,us2",
        markets="h2h,spreads",
        oddsFormat="american",
        dateFormat="iso",
    )
    events = odds
    if not events:
        events = await _get_json(
            session,
            f"/sports/{NFL_SPORT_KEY}/events",
            dateFormat="iso",
        )
    return [
        str(e["id"])
        for e in (events or [])
        if (e.get("commence_time") or "").startswith(day_iso)
    ]
