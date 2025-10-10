#!/usr/bin/env python3
"""Comprehensive NFL betting analytics and prediction pipeline.

This script ingests game, player, and odds data from the MySportsFeeds and The Odds
API services, persists the information to PostgreSQL, and then builds machine
learning models to predict player performance and game outcomes.

The workflow is designed to run incrementally: the first execution ingests all
available data for the configured seasons, while subsequent runs only request new
games and odds. The machine learning models incorporate contextual features such
as venue effects, day-of-week trends, officiating crews, weather, and team unit
strengths (rush/pass offense & defense) to deliver rich predictive insights that
can be used to identify profitable betting opportunities.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import math
import os
import re
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from requests import HTTPError
from requests.auth import HTTPBasicAuth
from requests.exceptions import JSONDecodeError as RequestsJSONDecodeError
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    func,
    select,
    text,
    inspect,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
# ==== BEGIN LINEUP + PANDAS PATCH HELPERS ===================================
def _is_effectively_empty_df(df: Optional[pd.DataFrame]) -> bool:
    if df is None:
        return True
    if not isinstance(df, pd.DataFrame):
        return True
    if df.empty:
        return True
    # all columns all-NA
    try:
        if df.shape[1] == 0:
            return True
        if all(df[col].isna().all() for col in df.columns):
            return True
    except Exception:
        pass
    return False

def safe_concat(frames: List[pd.DataFrame], **kwargs) -> pd.DataFrame:
    """Concat that ignores None/empty/all-NA frames to avoid FutureWarnings and dtype drift."""
    cleaned = [f for f in frames if not _is_effectively_empty_df(f)]
    if not cleaned:
        # Return an empty but stable DataFrame if everything is empty
        return pd.DataFrame()
    if len(cleaned) == 1:
        # Avoid returning a view into the input DataFrame which could be mutated upstream
        return cleaned[0].copy()
    return pd.concat(cleaned, **kwargs)

def coerce_boolean_mask(mask_like) -> pd.Series:
    """
    Robustly convert a mask with possible NA/object dtype to a clean boolean Series without FutureWarnings.
    """
    s = pd.Series(mask_like)
    # Try to preserve index if it's already a Series
    try:
        if hasattr(mask_like, "index"):
            s.index = mask_like.index
    except Exception:
        pass
    # Convert to pandas nullable boolean, fill, then to numpy bool
    s = s.astype("boolean").fillna(False)
    return s.astype(bool)

_POS_RE = re.compile(r"^(Offense|Defense|SpecialTeams)-([A-Za-z]+)(?:-(\d+))?$")

# Slots we want to project for offense; keep shallowest (lowest) depth per slot
_OFFENSE_KEEP = {
    "QB": 1,           # 1 QB
    "RB": 3,           # up to RB-3 (depth 1..3)
    "WR": 3,           # up to WR-3
    "TE": 2,           # up to TE-2 is fine
}

# Map MSF slot tokens to our POS
_SLOT_TO_POS = {
    "QB": "QB",
    "RB": "RB",
    "WR": "WR",
    "TE": "TE",
    # Explicit backfield/receiver aliases that occasionally appear without suffixes
    "HB": "RB",
    "FB": "RB",
    "TB": "RB",
    "SLOT": "WR",
    # OL/DEF/ST are ignored in player projections
}

_SLOT_PREFIX_MAP = {
    "QB": "QB",
    "RB": "RB",
    "HB": "RB",
    "FB": "RB",
    "TB": "RB",
    "WR": "WR",
    "TE": "TE",
    "SLOT": "WR",
}


def _slot_token_to_pos(token: Optional[str]) -> str:
    token_upper = (token or "").strip().upper()
    if not token_upper:
        return ""
    if token_upper in _SLOT_TO_POS:
        return _SLOT_TO_POS[token_upper]
    for prefix, canonical in _SLOT_PREFIX_MAP.items():
        if token_upper.startswith(prefix):
            return canonical
    return ""

def _parse_slot(slot: str) -> Tuple[str, Optional[int], Optional[str]]:
    """
    'Offense-WR-2' -> ('WR', 2)
    'Offense-QB-1' -> ('QB', 1)
    'Offense-TE-1' -> ('TE', 1)
    Unused/unknown returns ('', None)
    """
    m = _POS_RE.match(slot or "")
    if not m:
        return ("", None, None)
    side_token, token, depth = m.groups()
    pos = _slot_token_to_pos(token)
    d = int(depth) if depth and depth.isdigit() else None
    side = side_token.title() if side_token else None
    return (pos, d, side)

def _prefer_actual_then_expected(team_entry: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    """
    From a single team lineup block, return the best lineupPositions list and the section label.
    Prefers 'actual' if present and non-empty; else 'expected'; else [] with an empty label.
    """
    actual = (team_entry.get("actual") or {}).get("lineupPositions") or []
    if actual:
        return actual, "actual"
    expected = (team_entry.get("expected") or {}).get("lineupPositions") or []
    if expected:
        return expected, "expected"
    return [], ""

def build_lineups_df(msf_json: Dict[str, Any]) -> pd.DataFrame:
    """
    Build a clean lineup DF from MSF lineup JSON.
    - Prefers 'actual' but falls back to 'expected' (your case: NYG QB-1 Jaxson Dart / RB-1 Cam Skattebo).
    - Normalizes slots, keeps only QB/RB/WR/TE up to configured depth caps.
    - Collapses duplicates to shallowest depth per team/pos/player.
    Returns columns:
      [
          'team_id','team_abbr','pos','depth','side','slot','player_id','first','last',
          'full_name','player_team_abbr','playing_probability'
      ]
    """
    references = msf_json.get("references") or {}
    team_meta = {t.get("id"): t for t in references.get("teamReferences", []) or []}
    player_meta = {p.get("id"): p for p in references.get("playerReferences", []) or []}
    rows: List[Dict[str, Any]] = []

    for team_block in msf_json.get("teamLineups", []) or []:
        team = team_block.get("team") or {}
        team_id = team.get("id")
        abbr = team.get("abbreviation")
        positions, section_label = _prefer_actual_then_expected(team_block)

        for p in positions:
            slot = p.get("position")
            player = p.get("player") or {}
            if not player and p.get("playerId") is not None:
                player = player_meta.get(p.get("playerId"), {})
            pos, depth, side = _parse_slot(slot)
            if pos not in _OFFENSE_KEEP:
                continue
            if depth is None or depth > _OFFENSE_KEEP[pos]:
                continue
            player_id = player.get("id") or p.get("playerId")
            first = (player.get("firstName") or "").strip()
            last = (player.get("lastName") or "").strip()
            display = (player.get("displayName") or player.get("fullName") or "").strip()
            full_name = " ".join(part for part in [first, last] if part) or display
            if not full_name and player_id in player_meta:
                meta = player_meta.get(player_id) or {}
                first = first or (meta.get("firstName") or "").strip()
                last = last or (meta.get("lastName") or "").strip()
                display = display or (meta.get("displayName") or meta.get("fullName") or "")
                full_name = " ".join(part for part in [first, last] if part) or display.strip()
            if not full_name and player_id in (None, ""):
                # Without a name or identifier we cannot reconcile the player later.
                continue

            current_team_info = (
                player.get("currentTeam")
                or player.get("team")
                or (player_meta.get(player_id, {}) or {}).get("currentTeam")
                or {}
            )
            player_team_abbr = (
                current_team_info.get("abbreviation")
                or current_team_info.get("name")
                or ""
            )
            playing_probability = (
                p.get("playingProbability")
                or player.get("playingProbability")
                or (player_meta.get(player_id, {}) or {}).get("playingProbability")
            )
            rows.append({
                "team_id": team_id,
                "team_abbr": abbr,
                "pos": pos,
                "depth": depth,
                "side": side,
                "slot": slot,
                "player_id": player_id,
                "first": first,
                "last": last,
                "full_name": full_name,
                "player_team_abbr": player_team_abbr,
                "playing_probability": playing_probability,
                "source_section": section_label,
            })

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Deduplicate: keep shallowest depth per team/pos/player
    sort_cols = [col for col in ["team_id", "team_abbr", "pos", "player_id", "depth"] if col in df.columns]
    if sort_cols:
        df.sort_values(sort_cols, inplace=True)
    group_cols = [col for col in ["team_id", "pos", "player_id"] if col in df.columns]
    if group_cols:
        df = df.groupby(group_cols, as_index=False).first()

    # Also, per team/pos, keep at most the allowed number of depth slots
    df["rank_in_pos"] = df.groupby(["team_id", "pos"])["depth"].rank(method="first", ascending=True)
    df = df[df["rank_in_pos"] <= df["pos"].map(_OFFENSE_KEEP).fillna(0)]
    df.drop(columns=["rank_in_pos"], inplace=True)

    # Fill abbr from teamReferences if missing
    df["team_abbr"] = df.apply(
        lambda r: r["team_abbr"] or (team_meta.get(r["team_id"], {}).get("abbreviation")), axis=1
    )
    df["full_name"] = df.apply(
        lambda r: r["full_name"]
        or " ".join(part for part in [r.get("first", ""), r.get("last", "")] if part).strip(),
        axis=1,
    )
    df = df[df["full_name"].fillna("") != ""]
    return df.reset_index(drop=True)
# ==== END LINEUP + PANDAS PATCH HELPERS =====================================

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_PREFIX_NFL = "https://api.mysportsfeeds.com/v2.1/pull/nfl"
NFL_SEASONS = ["2024-regular", "2025-regular"]

NFL_API_USER = "4359aa1b-cc29-4647-a3e5-7314e2"
NFL_API_PASS = "MYSPORTSFEEDS"

ODDS_API_KEY = "5b6f0290e265c3329b3ed27897d79eaf"
ODDS_BASE = "https://api.the-odds-api.com/v4"
NFL_SPORT_KEY = "americanfootball_nfl"
ODDS_REGIONS = ["us"]
ODDS_FORMAT = "american"

DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# ---------------------------------------------------------------------------
# Static data helpers
# ---------------------------------------------------------------------------


TEAM_NAME_TO_ABBR = {
    "arizona cardinals": "ARI",
    "atlanta falcons": "ATL",
    "baltimore ravens": "BAL",
    "buffalo bills": "BUF",
    "carolina panthers": "CAR",
    "chicago bears": "CHI",
    "cincinnati bengals": "CIN",
    "cleveland browns": "CLE",
    "dallas cowboys": "DAL",
    "denver broncos": "DEN",
    "detroit lions": "DET",
    "green bay packers": "GB",
    "houston texans": "HOU",
    "indianapolis colts": "IND",
    "jacksonville jaguars": "JAX",
    "jacksonville jaguar": "JAX",
    "kansas city chiefs": "KC",
    "las vegas raiders": "LV",
    "oakland raiders": "LV",
    "los angeles chargers": "LAC",
    "la chargers": "LAC",
    "los angeles rams": "LA",
    "la rams": "LA",
    "miami dolphins": "MIA",
    "minnesota vikings": "MIN",
    "new england patriots": "NE",
    "new orleans saints": "NO",
    "new york giants": "NYG",
    "ny giants": "NYG",
    "new york jets": "NYJ",
    "ny jets": "NYJ",
    "philadelphia eagles": "PHI",
    "pittsburgh steelers": "PIT",
    "san francisco 49ers": "SF",
    "sf 49ers": "SF",
    "seattle seahawks": "SEA",
    "tampa bay buccaneers": "TB",
    "tennessee titans": "TEN",
    "washington commanders": "WAS",
    "washington football team": "WAS",
    "washington redskins": "WAS",
    "st. louis rams": "LA",
    "st louis rams": "LA",
}

TEAM_ABBR_CANONICAL = {
    "ARI": "arizona cardinals",
    "ATL": "atlanta falcons",
    "BAL": "baltimore ravens",
    "BUF": "buffalo bills",
    "CAR": "carolina panthers",
    "CHI": "chicago bears",
    "CIN": "cincinnati bengals",
    "CLE": "cleveland browns",
    "DAL": "dallas cowboys",
    "DEN": "denver broncos",
    "DET": "detroit lions",
    "GB": "green bay packers",
    "HOU": "houston texans",
    "IND": "indianapolis colts",
    "JAX": "jacksonville jaguars",
    "KC": "kansas city chiefs",
    "LV": "las vegas raiders",
    "LAC": "los angeles chargers",
    "LA": "los angeles rams",
    "MIA": "miami dolphins",
    "MIN": "minnesota vikings",
    "NE": "new england patriots",
    "NO": "new orleans saints",
    "NYG": "new york giants",
    "NYJ": "new york jets",
    "PHI": "philadelphia eagles",
    "PIT": "pittsburgh steelers",
    "SF": "san francisco 49ers",
    "SEA": "seattle seahawks",
    "TB": "tampa bay buccaneers",
    "TEN": "tennessee titans",
    "WAS": "washington commanders",
}

TEAM_ABBR_ALIASES = {
    "LA": "LA",
    "STL": "LA",
    "SD": "LAC",
}

TEAM_TIMEZONES = {
    "ARI": "America/Phoenix",
    "ATL": "America/New_York",
    "BAL": "America/New_York",
    "BUF": "America/New_York",
    "CAR": "America/New_York",
    "CHI": "America/Chicago",
    "CIN": "America/New_York",
    "CLE": "America/New_York",
    "DAL": "America/Chicago",
    "DEN": "America/Denver",
    "DET": "America/Detroit",
    "GB": "America/Chicago",
    "HOU": "America/Chicago",
    "IND": "America/Indiana/Indianapolis",
    "JAX": "America/New_York",
    "KC": "America/Chicago",
    "LV": "America/Los_Angeles",
    "LAC": "America/Los_Angeles",
    "LA": "America/Los_Angeles",
    "MIA": "America/New_York",
    "MIN": "America/Chicago",
    "NE": "America/New_York",
    "NO": "America/Chicago",
    "NYG": "America/New_York",
    "NYJ": "America/New_York",
    "PHI": "America/New_York",
    "PIT": "America/New_York",
    "SEA": "America/Los_Angeles",
    "SF": "America/Los_Angeles",
    "TB": "America/New_York",
    "TEN": "America/Chicago",
    "WAS": "America/New_York",
}

_NULL_TEAM_TOKENS = {"", "none", "null", "nan", "tbd", "tba", "n/a", "na", "--"}

TEAM_MASCOT_TO_ABBR = {
    "cardinals": "ARI",
    "falcons": "ATL",
    "ravens": "BAL",
    "bills": "BUF",
    "panthers": "CAR",
    "bears": "CHI",
    "bengals": "CIN",
    "browns": "CLE",
    "cowboys": "DAL",
    "broncos": "DEN",
    "lions": "DET",
    "packers": "GB",
    "texans": "HOU",
    "colts": "IND",
    "jaguars": "JAX",
    "chiefs": "KC",
    "raiders": "LV",
    "chargers": "LAC",
    "rams": "LA",
    "dolphins": "MIA",
    "vikings": "MIN",
    "patriots": "NE",
    "saints": "NO",
    "giants": "NYG",
    "jets": "NYJ",
    "eagles": "PHI",
    "steelers": "PIT",
    "49ers": "SF",
    "niners": "SF",
    "seahawks": "SEA",
    "buccaneers": "TB",
    "bucs": "TB",
    "titans": "TEN",
    "commanders": "WAS",
    "football team": "WAS",
}


def _sanitize_team_key(text: str) -> str:
    cleaned = []
    for ch in text.lower():
        if ch.isalnum() or ch.isspace():
            cleaned.append(ch)
    normalized = " ".join("".join(cleaned).split())
    return normalized


def normalize_team_abbr(value: Any) -> Optional[str]:
    """Convert free-form team descriptors into standard three-letter abbreviations."""

    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):  # type: ignore[arg-type]
        return None

    text = str(value).strip()
    if not text:
        return None

    lowered = text.lower().strip()
    if lowered in _NULL_TEAM_TOKENS:
        return None

    candidate = text.upper().replace(" ", "")
    if len(candidate) <= 4 and candidate.isalpha():
        if candidate in TEAM_ABBR_CANONICAL:
            return candidate
        if candidate in TEAM_ABBR_ALIASES:
            return TEAM_ABBR_ALIASES[candidate]
        # Some feeds already provide three-letter abbreviations with spaces
        spaced_candidate = " ".join(candidate)
        if spaced_candidate in TEAM_NAME_TO_ABBR:
            return TEAM_NAME_TO_ABBR[spaced_candidate]

    sanitized = _sanitize_team_key(text)
    if not sanitized:
        return None

    if sanitized in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[sanitized]

    sanitized_candidate = sanitized.replace(" ", "").upper()
    if sanitized_candidate in TEAM_ABBR_ALIASES:
        return TEAM_ABBR_ALIASES[sanitized_candidate]

    compact_key = sanitized.replace(" ", "")
    if compact_key in TEAM_NAME_TO_ABBR:
        return TEAM_NAME_TO_ABBR[compact_key]

    for abbr, canonical in TEAM_ABBR_CANONICAL.items():
        if sanitized == canonical:
            return abbr
        if canonical in sanitized:
            return abbr

    if sanitized in TEAM_MASCOT_TO_ABBR:
        return TEAM_MASCOT_TO_ABBR[sanitized]

    # As a last resort, try to map by taking the first letter of each token
    tokens = sanitized.split()
    if len(tokens) >= 2:
        initials = "".join(token[0] for token in tokens)
        if initials.upper() in TEAM_ABBR_CANONICAL:
            return initials.upper()

    if len(candidate) <= 4 and candidate.isalpha():
        return candidate

    return None


INJURY_STATUS_MAP = {
    "out": "out",
    "doubtful": "doubtful",
    "questionable": "questionable",
    "probable": "probable",
    "suspended": "suspended",
    "injured reserve": "out",
    "physically unable to perform": "out",
    "reserve/covid-19": "out",
    "covid-19": "out",
    "non-football injury": "out",
    "pup": "out",
    "ir": "out",
    "injury list": "out",
    "injury_list": "out",
    "injurylist": "out",
}

INJURY_OUT_KEYWORDS = [
    "injured reserve",
    "season-ending",
    "season ending",
    "out for season",
    "out indefinitely",
    "placed on ir",
    "on ir",
    "reserve/",
    "nfi",
    "pup",
    "physically unable to perform",
    "injury list",
    "injury_list",
    "injurylist",
]

INJURY_STATUS_PRIORITY = {
    "out": -1,
    "suspended": -1,
    "doubtful": 0,
    "questionable": 1,
    "probable": 2,
    "other": 1,
}

PRACTICE_STATUS_PRIORITY = {
    "full": 3,
    "limited": 2,
    "dnp": 0,
    "rest": 2,
    "available": 2,
}

PRACTICE_STATUS_ALIASES = {
    "fp": "full",
    "full practice": "full",
    "full participation": "full",
    "lp": "limited",
    "limited practice": "limited",
    "limited participation": "limited",
    "did not practice": "dnp",
    "did not participate": "dnp",
    "out": "dnp",
    "no practice": "dnp",
    "rest": "rest",
    "not injury related": "rest",
    "available": "available",
    "injury list": "dnp",
    "injury_list": "dnp",
    "injurylist": "dnp",
}

INACTIVE_INJURY_BUCKETS = {"out", "suspended"}

POSITION_ALIAS_MAP = {
    "HB": "RB",
    "TB": "RB",
    "FB": "RB",
    "SLOT": "WR",
    "FL": "WR",
    "SE": "WR",
    "X": "WR",
    "Z": "WR",
    "Y": "TE",
}

POSITION_PREFIX_MAP = {
    "QB": "QB",
    "RB": "RB",
    "HB": "RB",
    "FB": "RB",
    "TB": "RB",
    "WR": "WR",
    "TE": "TE",
}


TARGET_ALLOWED_POSITIONS: Dict[str, set[str]] = {
    "passing_yards": {"QB"},
    "passing_tds": {"QB"},
    "rushing_yards": {"QB", "RB", "HB", "FB"},
    "rushing_tds": {"QB", "RB", "HB", "FB"},
    "receiving_yards": {"RB", "HB", "FB", "WR", "TE"},
    "receptions": {"RB", "HB", "FB", "WR", "TE"},
    "receiving_tds": {"RB", "HB", "FB", "WR", "TE"},
}


LINEUP_STALENESS_DAYS = 7
LINEUP_MAX_AGE_BEFORE_GAME_DAYS = 21  # allow expected lineups up to 3 weeks old relative to kickoff


_NAME_PUNCT_RE = re.compile(r"[^\w\s]")
_NAME_SPACE_RE = re.compile(r"\s+")


def robust_player_name_key(value: Any) -> str:
    """Generate a resilient lowercase key for player name matching."""

    if value is None:
        text = ""
    else:
        text = str(value)

    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    lowered = normalized.lower()
    without_punct = _NAME_PUNCT_RE.sub(" ", lowered)
    collapsed = _NAME_SPACE_RE.sub(" ", without_punct).strip()
    return collapsed


def normalize_player_name(value: Any) -> str:
    """Return a lowercase, punctuation-free representation of a player's name."""

    if not isinstance(value, str):
        return ""

    normalized = unicodedata.normalize("NFKD", value)
    normalized = normalized.replace(",", " ")
    cleaned = [ch for ch in normalized if ch.isalpha() or ch.isspace()]
    collapsed = " ".join("".join(cleaned).lower().split())
    return collapsed


def normalize_position(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).upper().strip()
    if not text:
        return ""
    text = POSITION_ALIAS_MAP.get(text, text)
    for prefix, canonical in POSITION_PREFIX_MAP.items():
        if text.startswith(prefix):
            return canonical
    return text


def normalize_practice_status(value: Any) -> str:
    text = str(value or "").lower().strip()
    if not text:
        return "available"
    if re.search(r"\b(ir|injured reserve|pup|nfi|reserve)\b", text):
        return "dnp"
    if text in PRACTICE_STATUS_ALIASES:
        text = PRACTICE_STATUS_ALIASES[text]
    else:
        for keyword, canonical in (
            ("full", "full"),
            ("limited", "limited"),
            ("dnp", "dnp"),
            ("did not practice", "dnp"),
            ("rest", "rest"),
        ):
            if keyword in text:
                text = canonical
                break
    if text not in PRACTICE_STATUS_PRIORITY:
        return "available"
    return text


_PLAYING_PROBABILITY_ALIASES = {
    "prob": "probable",
    "probable": "probable",
    "likely": "probable",
    "expected": "probable",
    "game-time decision": "questionable",
    "gtd": "questionable",
    "game time decision": "questionable",
    "game-time": "questionable",
    "uncertain": "questionable",
    "na": "other",
}


def interpret_playing_probability(value: Any) -> Tuple[str, str]:
    """Return (status_bucket, practice_status) derived from MSF playingProbability labels."""

    text_raw = str(value or "").strip().lower()
    if not text_raw:
        return "other", "available"

    cleaned = re.sub(r"[^a-z\s]", " ", text_raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return "other", "available"

    canonical = _PLAYING_PROBABILITY_ALIASES.get(cleaned, cleaned)

    keyword_rules = [
        ("suspend", ("suspended", "dnp")),
        ("doubt", ("doubtful", "limited")),
        ("question", ("questionable", "limited")),
        ("inactive", ("out", "dnp")),
        ("out", ("out", "dnp")),
        ("probable", ("probable", "available")),
        ("likely", ("probable", "available")),
        ("expect", ("probable", "available")),
        ("available", ("other", "full")),
        ("active", ("other", "full")),
    ]

    for keyword, outcome in keyword_rules:
        if keyword in canonical:
            return outcome

    return "other", "available"


def normalize_injury_status(value: Any) -> str:
    text = str(value or "").lower().strip()
    if not text:
        return "other"
    if text in INJURY_STATUS_MAP:
        return INJURY_STATUS_MAP[text]
    if re.search(r"\b(ir|pup|nfi|reserve)\b", text):
        return "out"
    if "questionable" in text:
        return "questionable"
    if "doubtful" in text:
        return "doubtful"
    if "probable" in text:
        return "probable"
    if "suspend" in text:
        return "suspended"
    for keyword in INJURY_OUT_KEYWORDS:
        if keyword in text:
            return "out"
    return "other"


def compute_injury_bucket(status: Any, description: Any = None) -> str:
    """Derive a canonical availability bucket from status and free-form notes."""

    bucket = normalize_injury_status(status)
    if bucket != "out":
        desc_text = str(description or "").lower().strip()
        for keyword in INJURY_OUT_KEYWORDS:
            if keyword in desc_text:
                bucket = "out"
                break
    return bucket


def parse_depth_rank(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().lower()
    if not text:
        return np.nan

    alias_map = {
        "starter": 1,
        "start": 1,
        "first": 1,
        "1st": 1,
        "qb1": 1,
        "rb1": 1,
        "wr1": 1,
        "te1": 1,
        "second": 2,
        "2nd": 2,
        "qb2": 2,
        "rb2": 2,
        "wr2": 2,
        "third": 3,
        "3rd": 3,
        "qb3": 3,
        "rb3": 3,
        "wr3": 3,
    }
    if text in alias_map:
        return float(alias_map[text])

    digits = "".join(ch for ch in text if ch.isdigit())
    if digits:
        try:
            return float(int(digits))
        except ValueError:
            return np.nan

    return np.nan


def ensure_lineup_players_in_latest(
    latest_players: pd.DataFrame, lineup_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Add synthetic placeholder rows for lineup players missing from feature data."""

    if lineup_df is None or lineup_df.empty:
        return latest_players

    working = latest_players.copy()
    if "team" not in working.columns:
        working["team"] = np.nan
    if "position" not in working.columns:
        working["position"] = np.nan
    if "player_name" not in working.columns:
        working["player_name"] = ""
    if "player_name_norm" not in working.columns:
        working["player_name_norm"] = working["player_name"].map(normalize_player_name)

    if "__pname_key" not in working.columns:
        working["__pname_key"] = working["player_name"].map(robust_player_name_key)
    else:
        working["__pname_key"] = working["__pname_key"].fillna("")
        missing_key_mask = working["__pname_key"] == ""
        if missing_key_mask.any():
            working.loc[missing_key_mask, "__pname_key"] = working.loc[
                missing_key_mask, "player_name"
            ].map(robust_player_name_key)

    template_columns = list(working.columns)

    lineup = lineup_df.copy()
    lineup["team"] = lineup["team"].apply(normalize_team_abbr)
    lineup["position"] = lineup["position"].apply(normalize_position)
    if "base_pos" in lineup.columns:
        lineup["base_pos"] = lineup["base_pos"].apply(normalize_position)
    else:
        lineup["base_pos"] = lineup["position"]

    if "__pname_key" not in lineup.columns:
        lineup["__pname_key"] = ""
    name_seed = (
        lineup.get("first_name", "").fillna("") + " " + lineup.get("last_name", "").fillna("")
    ).str.strip()
    fallback_name = lineup.get("player_name", "").fillna("")
    lineup["__pname_key"] = lineup["__pname_key"].fillna("")
    needs_key = lineup["__pname_key"] == ""
    lineup.loc[needs_key, "__pname_key"] = name_seed.where(name_seed != "", fallback_name)[
        needs_key
    ].map(robust_player_name_key)
    lineup["__pname_key"] = lineup["__pname_key"].fillna("")
    lineup = lineup[lineup["__pname_key"] != ""]

    if "side" in lineup.columns:
        lineup = lineup[
            lineup["side"].fillna("").str.lower().isin({"offense", ""})
        ]

    lineup = lineup[lineup["base_pos"].isin({"QB", "RB", "WR", "TE"})]
    if lineup.empty:
        return working

    existing_keys = set(
        zip(
            working["team"].fillna(""),
            working["position"].fillna(""),
            working["__pname_key"].fillna(""),
        )
    )

    feature_defaults = {
        "snap_share": 0.35,
        "routes_run": 15.0,
        "targets_per_g": 2.0,
        "rush_att_per_g": 6.0,
        "games_last3": 1.0,
    }

    additions: List[Dict[str, Any]] = []

    for _, lineup_row in lineup.iterrows():
        team = lineup_row.get("team")
        base_pos = lineup_row.get("base_pos")
        pname_key = lineup_row.get("__pname_key")
        if not team or not base_pos or not pname_key:
            continue
        key = (team, base_pos, pname_key)
        if key in existing_keys:
            continue

        player_name = lineup_row.get("player_name") or ""
        if not player_name:
            first = lineup_row.get("first_name", "")
            last = lineup_row.get("last_name", "")
            player_name = " ".join(part for part in [first, last] if part)

        placeholder = {col: np.nan for col in template_columns}
        placeholder["team"] = team
        placeholder["position"] = base_pos
        placeholder["player_name"] = player_name
        placeholder["player_name_norm"] = normalize_player_name(player_name)
        placeholder["__pname_key"] = pname_key

        raw_player_id = lineup_row.get("player_id")
        if isinstance(raw_player_id, str) and raw_player_id:
            placeholder["player_id"] = raw_player_id
        else:
            placeholder["player_id"] = f"lineup_{team}_{pname_key}"

        depth_rank = lineup_row.get("rank")
        placeholder["depth_rank"] = depth_rank if depth_rank not in (None, "") else 1

        lineup_status = lineup_row.get("status_bucket")
        lineup_practice = lineup_row.get("practice_status")
        if lineup_status:
            status_bucket = normalize_injury_status(lineup_status)
            practice_status = normalize_practice_status(lineup_practice)
        else:
            status_bucket, practice_status = interpret_playing_probability(
                lineup_row.get("playing_probability")
            )
            status_bucket = normalize_injury_status(status_bucket)
            practice_status = normalize_practice_status(practice_status)

        placeholder["status_bucket"] = status_bucket
        placeholder["practice_status"] = practice_status
        if "injury_priority" in placeholder:
            placeholder["injury_priority"] = INJURY_STATUS_PRIORITY.get(status_bucket, 1)
        if "practice_priority" in placeholder:
            placeholder["practice_priority"] = PRACTICE_STATUS_PRIORITY.get(
                practice_status, PRACTICE_STATUS_PRIORITY.get("available", 1)
            )

        if "_lineup_entry" in placeholder:
            placeholder["_lineup_entry"] = True
        if "source" in placeholder:
            placeholder["source"] = "msf-lineup"

        updated_at = lineup_row.get("updated_at")
        game_start = lineup_row.get("game_start")
        if "updated_at" in placeholder:
            placeholder["updated_at"] = updated_at
        if "game_start" in placeholder:
            placeholder["game_start"] = game_start

        if "first_name" in placeholder:
            placeholder["first_name"] = lineup_row.get("first_name", "")
        if "last_name" in placeholder:
            placeholder["last_name"] = lineup_row.get("last_name", "")

        for col, default_val in feature_defaults.items():
            if col in placeholder:
                placeholder[col] = default_val

        for col in template_columns:
            if col.startswith("season_"):
                placeholder[col] = 0.0

        additions.append(placeholder)
        existing_keys.add(key)

    if additions:
        addition_df = pd.DataFrame(additions)
        working = safe_concat([working, addition_df], ignore_index=True, sort=False)

    return working


# ---------------------------------------------------------------------------
# Supplemental data loading
# ---------------------------------------------------------------------------


class SupplementalDataLoader:
    """Loads optional injury, depth chart, advanced metric, and weather feeds."""

    def __init__(self, config: NFLConfig):
        self.injury_records = self._load_records(config.injury_report_path)
        self.depth_chart_records = self._load_records(config.depth_chart_path)
        self.advanced_records = self._load_records(config.advanced_metrics_path)
        self.weather_records = self._load_records(config.weather_forecast_path)

        self.injuries_by_game = self._index_records(self.injury_records, "game_id")
        self.injuries_by_team = self._index_records(self.injury_records, "team")
        self.depth_by_team = self._index_records(self.depth_chart_records, "team")
        self.weather_by_game = self._index_records(self.weather_records, "game_id")
        self.advanced_by_key: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
        for record in self.advanced_records:
            season = record.get("season")
            week = record.get("week")
            team = normalize_team_abbr(record.get("team"))
            if season and week is not None and team:
                self.advanced_by_key[(str(season), int(week), team)] = record

    @staticmethod
    def _load_records(path: Optional[str]) -> List[Dict[str, Any]]:
        if not path:
            return []
        file_path = Path(path)
        if not file_path.exists():
            logging.warning("Supplemental data file %s not found", file_path)
            return []
        try:
            if file_path.suffix.lower() in {".json", ".geojson"}:
                with file_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    if "items" in payload and isinstance(payload["items"], list):
                        return [dict(item) for item in payload["items"]]
                    if "data" in payload and isinstance(payload["data"], list):
                        return [dict(item) for item in payload["data"]]
                    return [dict(payload)]
                if isinstance(payload, list):
                    return [dict(item) for item in payload]
                logging.warning("Unsupported JSON format in %s", file_path)
                return []
            frame = pd.read_csv(file_path)
            return frame.to_dict("records")
        except Exception:  # pragma: no cover - defensive logging
            logging.exception("Unable to load supplemental data from %s", file_path)
            return []

    @staticmethod
    def _index_records(records: List[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
        index: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            raw_value = record.get(key)
            if raw_value is None:
                continue
            if key == "team":
                normalized = normalize_team_abbr(raw_value)
            else:
                normalized = str(raw_value)
            if not normalized:
                continue
            index.setdefault(normalized, []).append(record)
        return index

    def injuries_for_game(
        self, game_id: str, home_team: Optional[str], away_team: Optional[str]
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        game_records = list(self.injuries_by_game.get(str(game_id), []))
        if not game_records:
            for team in (home_team, away_team):
                if not team:
                    continue
                team_records = self.injuries_by_team.get(normalize_team_abbr(team), [])
                game_records.extend(team_records)

        normalized_rows: List[Dict[str, Any]] = []
        for record in game_records:
            team = normalize_team_abbr(record.get("team"))
            player_name = record.get("player_name") or record.get("name")
            status = record.get("status")
            practice_status = record.get("practice_status") or record.get("practice")
            description = record.get("description") or record.get("details")
            report_time = record.get("report_time") or record.get("updated_at")
            position = normalize_position(
                record.get("position")
                or record.get("primary_position")
                or record.get("pos")
            )
            normalized_rows.append(
                {
                    "injury_id": record.get("injury_id")
                    or record.get("id")
                    or uuid.uuid4().hex,
                    "game_id": str(game_id),
                    "team": team,
                    "player_name": player_name,
                    "status": status,
                    "practice_status": practice_status,
                    "description": description,
                    "report_time": parse_dt(report_time) if report_time else None,
                    "position": position,
                }
            )

        summary_parts = [
            f"{row['player_name']}({row['status']})"
            for row in normalized_rows
            if row.get("player_name") and row.get("status")
        ]
        summary: Optional[str] = None
        if summary_parts:
            preview = summary_parts[:6]
            summary = ", ".join(preview)
            remaining = len(summary_parts) - len(preview)
            if remaining > 0:
                summary += f" +{remaining} more"
        return normalized_rows, summary

    def depth_chart_rows(self, team: str) -> List[Dict[str, Any]]:
        normalized_team = normalize_team_abbr(team)
        records = self.depth_by_team.get(normalized_team, [])
        rows: List[Dict[str, Any]] = []
        for record in records:
            rows.append(
                {
                    "depth_id": record.get("depth_id")
                    or record.get("id")
                    or uuid.uuid4().hex,
                    "team": normalized_team,
                    "position": record.get("position"),
                    "player_id": str(record.get("player_id") or record.get("id") or ""),
                    "player_name": record.get("player_name") or record.get("name"),
                    "rank": record.get("rank") or record.get("depth") or record.get("order"),
                    "updated_at": parse_dt(record.get("updated_at"))
                    if record.get("updated_at")
                    else parse_dt(record.get("timestamp")),
                }
            )
        return rows

    def advanced_metrics(
        self, season: Optional[str], week: Optional[int], team: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if not season or week is None or not team:
            return None
        return self.advanced_by_key.get((str(season), int(week), normalize_team_abbr(team)))

    def weather_override(self, game_id: str) -> Optional[Dict[str, Any]]:
        records = self.weather_by_game.get(str(game_id), [])
        if not records:
            return None
        latest = max(records, key=lambda rec: rec.get("updated_at") or "")
        output = dict(latest)
        if "temperature_f" not in output and "temperature" in output:
            output["temperature_f"] = NFLIngestor._extract_temperature_fahrenheit(
                output.get("temperature")
            )
        return output

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


def default_now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def parse_dt(value: str) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


_MSF_ABBR_FIX = {
    "JAC": "JAX",
    "LA": "LA",
    "WFT": "WAS",
    "WSH": "WAS",
    "ARZ": "ARI",
    "OAK": "LV",
    "SD": "LAC",
}


def _msf_team_abbr(abbr: Optional[str]) -> Optional[str]:
    if not abbr:
        return None
    value = abbr.strip().upper()
    return _MSF_ABBR_FIX.get(value, value)


_HOME_TZ = {
    "LA": "America/Los_Angeles",
    "LAC": "America/Los_Angeles",
    "LV": "America/Los_Angeles",
    "SF": "America/Los_Angeles",
    "SEA": "America/Los_Angeles",
    "ARI": "America/Phoenix",
    "DEN": "America/Denver",
    "CHI": "America/Chicago",
    "DAL": "America/Chicago",
    "GB": "America/Chicago",
    "HOU": "America/Chicago",
    "KC": "America/Chicago",
    "MIN": "America/Chicago",
    "NO": "America/Chicago",
    "TB": "America/Chicago",
    "TEN": "America/Chicago",
    "ATL": "America/New_York",
    "BAL": "America/New_York",
    "BUF": "America/New_York",
    "CAR": "America/New_York",
    "CIN": "America/New_York",
    "CLE": "America/New_York",
    "DET": "America/New_York",
    "IND": "America/New_York",
    "JAX": "America/New_York",
    "MIA": "America/New_York",
    "NE": "America/New_York",
    "NYG": "America/New_York",
    "NYJ": "America/New_York",
    "PHI": "America/New_York",
    "PIT": "America/New_York",
    "WAS": "America/New_York",
}


def _home_local_game_date(utc_start: dt.datetime, home_abbr: str) -> dt.date:
    canonical = TEAM_ABBR_ALIASES.get(home_abbr, home_abbr)
    tz_name = _HOME_TZ.get(canonical) or _HOME_TZ.get(home_abbr)
    if not tz_name:
        logging.warning(
            "Unknown home timezone for %s; defaulting to America/New_York",
            home_abbr,
        )
        tz = ZoneInfo("America/New_York")
    else:
        tz = ZoneInfo(tz_name)
    kickoff = utc_start
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=ZoneInfo("UTC"))
    else:
        kickoff = kickoff.astimezone(ZoneInfo("UTC"))
    return kickoff.astimezone(tz).date()


def _nfl_season_slug_for_start(start_time_utc: Optional[dt.datetime]) -> Optional[str]:
    if not start_time_utc:
        return None
    kickoff = start_time_utc
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=dt.timezone.utc)
    year = kickoff.year
    if kickoff.month < 8:
        start_year, end_year = year - 1, year
    else:
        start_year, end_year = year, year + 1
    return f"{start_year}-{end_year}-regular"


def _yyyy_mm_dd_from_utc(start_time_utc: dt.datetime) -> str:
    kickoff = start_time_utc
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=dt.timezone.utc)
    d = kickoff.date()
    return f"{d.year:04d}{d.month:02d}{d.day:02d}"


_POS_RE = re.compile(r"^([A-Za-z]+)-([A-Za-z]+)-(\d+)$", re.IGNORECASE)


def split_lineup_slot(slot: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """Return (side, base position, rank) extracted from lineup slot strings."""

    match = _POS_RE.match(slot or "")
    if not match:
        return None, None, None
    side_raw, base_raw, rank_raw = match.groups()
    side = side_raw.title()
    base_pos = base_raw.upper()
    try:
        rank_val = int(rank_raw)
    except ValueError:
        rank_val = None
    return side, base_pos, rank_val


def _canon_pos_and_rank(lineup_position: str) -> Tuple[Optional[str], Optional[int]]:
    _, base_pos, rank_val = split_lineup_slot(lineup_position)
    if base_pos not in {"QB", "RB", "WR", "TE"}:
        return None, None
    return base_pos, rank_val


def _prefer_actual(team_block: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    actual = (team_block.get("actual") or {}).get("lineupPositions") or []
    if actual:
        return actual, "actual"
    expected = (team_block.get("expected") or {}).get("lineupPositions") or []
    if expected:
        return expected, "expected"
    return [], ""


def _build_msf_lineup_url(
    start_time_utc: dt.datetime, away_abbr: str, home_abbr: str
) -> Optional[str]:
    season_slug = _nfl_season_slug_for_start(start_time_utc)
    if not season_slug:
        return None
    away_norm = _msf_team_abbr(away_abbr)
    home_norm = _msf_team_abbr(home_abbr)
    if not away_norm or not home_norm:
        return None
    local_date = _home_local_game_date(start_time_utc, home_norm)
    game_date = f"{local_date.year:04d}{local_date.month:02d}{local_date.day:02d}"
    return (
        f"https://api.mysportsfeeds.com/v2.1/pull/nfl/{season_slug}/games/"
        f"{game_date}-{away_norm}-{home_norm}/lineup.json"
    )


def _http_get_with_retry(
    url: str,
    auth: HTTPBasicAuth,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    max_tries: int = 3,
    backoff: float = 0.8,
) -> Optional[requests.Response]:
    last_exc: Optional[Exception] = None
    for attempt in range(max_tries):
        try:
            response = requests.get(
                url,
                auth=auth,
                params=params,
                headers=headers,
                timeout=15,
            )
            if response.status_code in {200, 204}:
                return response
            if response.status_code == 404:
                time.sleep(backoff)
                return response
            logging.warning(
                "MSF lineup GET %s -> %s; try %d/%d",
                url,
                response.status_code,
                attempt + 1,
                max_tries,
            )
            time.sleep(backoff * (attempt + 1))
        except Exception as exc:  # pragma: no cover - network errors
            last_exc = exc
            logging.warning(
                "MSF lineup GET exception on %s: %s (try %d/%d)",
                url,
                exc,
                attempt + 1,
                max_tries,
            )
            time.sleep(backoff * (attempt + 1))
    if last_exc is not None:
        logging.exception("MSF lineup GET failed after retries: %s", last_exc)
    return None


def _extract_lineup_rows(json_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    lineup_df = build_lineups_df(json_obj)
    if lineup_df.empty:
        return []

    results: List[Dict[str, Any]] = []
    for record in lineup_df.to_dict(orient="records"):
        team_abbr = record.get("team_abbr")
        position = record.get("pos")
        depth = record.get("depth")
        if not team_abbr or not position:
            continue
        player_name = record.get("full_name") or ""
        first = record.get("first") or ""
        last = record.get("last") or ""
        if not player_name:
            player_name = " ".join(part for part in [first, last] if part)
        if not player_name and not record.get("player_id"):
            continue

        pname_source = player_name or " ".join(part for part in [first, last] if part)
        status_bucket, practice_status = interpret_playing_probability(
            record.get("playing_probability")
        )
        results.append(
            {
                "team": _msf_team_abbr(team_abbr),
                "player_id": str(record.get("player_id") or ""),
                "player_name": player_name,
                "first_name": first,
                "last_name": last,
                "position": position,
                "base_pos": position,
                "side": record.get("side"),
                "rank": depth,
                "source_section": record.get("source_section") or "actual",
                "player_team": _msf_team_abbr(record.get("player_team_abbr")),
                "playing_probability": record.get("playing_probability"),
                "status_bucket": status_bucket,
                "practice_status": practice_status,
                "slot": record.get("slot"),
                "__pname_key": robust_player_name_key(pname_source),
            }
        )
    return results


@dataclasses.dataclass
class NFLConfig:
    pg_user: str = os.getenv("PGUSER", "josh")
    pg_password: str = os.getenv("PGPASSWORD", "password")
    pg_host: str = os.getenv("PGHOST", "localhost")
    pg_port: str = os.getenv("PGPORT", "5432")
    pg_database: str = os.getenv("PGDATABASE", "nfl")

    seasons: Tuple[str, ...] = tuple(NFL_SEASONS)
    log_level: str = DEFAULT_LOG_LEVEL
    injury_report_path: Optional[str] = os.getenv("NFL_INJURY_PATH")
    depth_chart_path: Optional[str] = os.getenv("NFL_DEPTH_PATH")
    advanced_metrics_path: Optional[str] = os.getenv("NFL_ADVANCED_PATH")
    weather_forecast_path: Optional[str] = os.getenv("NFL_FORECAST_PATH")
    respect_lineups: bool = True

    @property
    def pg_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )


# ---------------------------------------------------------------------------
# Database schema & helpers
# ---------------------------------------------------------------------------


class NFLDatabase:
    """Encapsulates PostgreSQL persistence for NFL data."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.meta = MetaData()
        self._define_tables()
        self.meta.create_all(self.engine)
        self._apply_schema_upgrades()

    def _apply_schema_upgrades(self) -> None:
        """Ensure newly introduced columns exist on already-initialized tables."""

        inspector = inspect(self.engine)
        try:
            table_names = set(inspector.get_table_names())
        except Exception:
            table_names = set()

        try:
            game_columns = {col["name"] for col in inspector.get_columns("nfl_games")}
        except Exception:  # pragma: no cover - defensive fallback if table missing
            game_columns = set()

        statements: List[str] = []
        if "wind_mph" not in game_columns:
            statements.append("ALTER TABLE nfl_games ADD COLUMN IF NOT EXISTS wind_mph DOUBLE PRECISION")
        if "humidity" not in game_columns:
            statements.append("ALTER TABLE nfl_games ADD COLUMN IF NOT EXISTS humidity DOUBLE PRECISION")
        if "injury_summary" not in game_columns:
            statements.append("ALTER TABLE nfl_games ADD COLUMN IF NOT EXISTS injury_summary TEXT")

        if "nfl_game_rosters" in table_names:
            try:
                roster_columns = {col["name"] for col in inspector.get_columns("nfl_game_rosters")}
            except Exception:
                roster_columns = set()
            for coldef in [
                "ADD COLUMN IF NOT EXISTS depth_rank INTEGER",
                "ADD COLUMN IF NOT EXISTS is_starter INTEGER",
                "ADD COLUMN IF NOT EXISTS source TEXT",
                "ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ",
                "ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ",
            ]:
                statements.append(f"ALTER TABLE nfl_game_rosters {coldef}")

        if "nfl_depth_charts" in table_names:
            try:
                depth_columns = {col["name"] for col in inspector.get_columns("nfl_depth_charts")}
            except Exception:
                depth_columns = set()
            if "source" not in depth_columns:
                statements.append(
                    "ALTER TABLE nfl_depth_charts ADD COLUMN IF NOT EXISTS source TEXT"
                )

        try:
            injury_columns = {col["name"] for col in inspector.get_columns("nfl_injury_reports")}
        except Exception:  # pragma: no cover - table may not exist yet
            injury_columns = set()
        if "position" not in injury_columns:
            statements.append(
                "ALTER TABLE nfl_injury_reports ADD COLUMN IF NOT EXISTS position TEXT"
            )

        if not statements:
            return

        with self.engine.begin() as conn:
            for statement in statements:
                conn.execute(text(statement))

    def _define_tables(self) -> None:
        self.games = Table(
            "nfl_games",
            self.meta,
            Column("game_id", String, primary_key=True),
            Column("season", String, nullable=False),
            Column("week", Integer),
            Column("start_time", DateTime(timezone=True)),
            Column("venue", String),
            Column("city", String),
            Column("state", String),
            Column("country", String),
            Column("surface", String),
            Column("day_of_week", String),
            Column("referee", String),
            Column("temperature_f", Float),
            Column("weather_conditions", String),
            Column("wind_mph", Float),
            Column("humidity", Float),
            Column("injury_summary", String),
            Column("home_team", String),
            Column("away_team", String),
            Column("home_score", Integer),
            Column("away_score", Integer),
            Column("status", String),
            Column("home_moneyline", Float),
            Column("away_moneyline", Float),
            Column("home_implied_prob", Float),
            Column("away_implied_prob", Float),
            Column("odds_updated", DateTime(timezone=True)),
            Column("ingested_at", DateTime(timezone=True), default=default_now_utc),
        )

        self.player_stats = Table(
            "nfl_player_stats",
            self.meta,
            Column("game_id", String, nullable=False),
            Column("player_id", String, nullable=False),
            Column("player_name", String),
            Column("team", String),
            Column("position", String),
            Column("rushing_attempts", Float),
            Column("rushing_yards", Float),
            Column("rushing_tds", Float),
            Column("receiving_targets", Float),
            Column("receptions", Float),
            Column("receiving_yards", Float),
            Column("receiving_tds", Float),
            Column("passing_attempts", Float),
            Column("passing_completions", Float),
            Column("passing_yards", Float),
            Column("passing_tds", Float),
            Column("fantasy_points", Float),
            Column("snap_count", Float),
            Column("ingested_at", DateTime(timezone=True), default=default_now_utc),
            UniqueConstraint("game_id", "player_id", name="uq_player_game"),
        )

        # NEW: per-game roster derived from MSF lineup.json
        self.game_rosters = Table(
            "nfl_game_rosters",
            self.meta,
            Column("game_id", String, nullable=False),
            Column("team", String, nullable=False),
            Column("player_id", String),
            Column("player_name", String, nullable=False),
            Column("position", String, nullable=False),
            Column("depth_rank", Integer),
            Column("is_starter", Integer),
            Column("source", String, nullable=False),
            Column("updated_at", DateTime(timezone=True), default=default_now_utc),
            Column("ingested_at", DateTime(timezone=True), default=default_now_utc),
            UniqueConstraint("game_id", "team", "player_id", "player_name", name="uq_game_roster"),
        )

        self.team_unit_ratings = Table(
            "nfl_team_unit_ratings",
            self.meta,
            Column("season", String, nullable=False),
            Column("team", String, nullable=False),
            Column("week", Integer, nullable=False),
            Column("offense_pass_rating", Float),
            Column("offense_rush_rating", Float),
            Column("defense_pass_rating", Float),
            Column("defense_rush_rating", Float),
            Column("updated_at", DateTime(timezone=True), default=default_now_utc),
            UniqueConstraint("season", "team", "week", name="uq_team_week"),
        )

        self.model_predictions = Table(
            "nfl_predictions",
            self.meta,
            Column("prediction_id", String, primary_key=True),
            Column("game_id", String, nullable=False),
            Column("entity_type", String, nullable=False),
            Column("entity_id", String, nullable=False),
            Column("prediction_target", String, nullable=False),
            Column("prediction_value", Float),
            Column("model_version", String),
            Column("features", JSON),
            Column("created_at", DateTime(timezone=True), default=default_now_utc),
        )

        self.injury_reports = Table(
            "nfl_injury_reports",
            self.meta,
            Column("injury_id", String, primary_key=True),
            Column("game_id", String),
            Column("team", String, nullable=False),
            Column("player_name", String),
            Column("position", String),
            Column("status", String),
            Column("practice_status", String),
            Column("description", String),
            Column("report_time", DateTime(timezone=True)),
            Column("ingested_at", DateTime(timezone=True), default=default_now_utc),
        )

        self.depth_charts = Table(
            "nfl_depth_charts",
            self.meta,
            Column("depth_id", String, primary_key=True),
            Column("team", String, nullable=False),
            Column("position", String, nullable=False),
            Column("player_id", String),
            Column("player_name", String),
            Column("rank", Integer),
            Column("source", String),
            Column("updated_at", DateTime(timezone=True)),
            Column("ingested_at", DateTime(timezone=True), default=default_now_utc),
        )

        self.team_advanced_metrics = Table(
            "nfl_team_advanced_metrics",
            self.meta,
            Column("metric_id", String, primary_key=True),
            Column("season", String, nullable=False),
            Column("week", Integer, nullable=False),
            Column("team", String, nullable=False),
            Column("pace_seconds_per_play", Float),
            Column("offense_epa", Float),
            Column("defense_epa", Float),
            Column("offense_success_rate", Float),
            Column("defense_success_rate", Float),
            Column("travel_penalty", Float),
            Column("rest_penalty", Float),
            Column("weather_adjustment", Float),
            Column("created_at", DateTime(timezone=True), default=default_now_utc),
            UniqueConstraint("season", "week", "team", name="uq_adv_metrics_team_week"),
        )

        self.model_backtests = Table(
            "nfl_model_backtests",
            self.meta,
            Column("run_id", String, nullable=False),
            Column("model_name", String, nullable=False),
            Column("metric_name", String, nullable=False),
            Column("metric_value", Float, nullable=False),
            Column("sample_size", Integer),
            Column("created_at", DateTime(timezone=True), default=default_now_utc),
        )

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    def upsert_rows(
        self,
        table: Table,
        rows: Iterable[Dict[str, Any]],
        conflict_cols: List[str],
        update_columns: Optional[Iterable[str]] = None,
    ) -> None:
        rows_list = list(rows)
        if not rows_list:
            return

        table_columns = set(table.c.keys())
        filtered_rows: List[Dict[str, Any]] = []
        for row in rows_list:
            if not isinstance(row, dict):
                continue
            filtered = {k: v for k, v in row.items() if k in table_columns}
            if filtered:
                filtered_rows.append(filtered)

        if not filtered_rows:
            return

        stmt = insert(table).values(filtered_rows)
        if update_columns is None:
            update_cols = {
                col.name: stmt.excluded[col.name]
                for col in table.columns
                if col.name not in conflict_cols
            }
        else:
            valid_columns = {
                col
                for col in update_columns
                if col in table.c.keys() and col not in conflict_cols
            }
            update_cols = {col: stmt.excluded[col] for col in valid_columns}

        if update_cols:
            stmt = stmt.on_conflict_do_update(index_elements=conflict_cols, set_=update_cols)
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=conflict_cols)
        try:
            with self.engine.begin() as conn:
                conn.execute(stmt)
        except SQLAlchemyError:
            logging.exception("Failed to upsert rows into %s", table.name)
            raise

    def upsert_game_rosters(self, rows: Iterable[Dict[str, Any]]) -> None:
        self.upsert_rows(self.game_rosters, list(rows), ["game_id", "team", "player_id", "player_name"])

    def fetch_game_roster(self, game_id: str) -> pd.DataFrame:
        with self.engine.begin() as conn:
            q = select(self.game_rosters).where(self.game_rosters.c.game_id == str(game_id))
            rows = conn.execute(q).mappings().all()
        if rows:
            return pd.DataFrame(rows)
        return pd.DataFrame(
            columns=[
                "game_id",
                "team",
                "player_id",
                "player_name",
                "position",
                "depth_rank",
                "is_starter",
                "source",
                "updated_at",
                "ingested_at",
            ]
        )

    def fetch_existing_game_ids(self) -> set[str]:
        with self.engine.begin() as conn:
            rows = conn.execute(select(self.games.c.game_id)).fetchall()
        return {row[0] for row in rows}

    def fetch_games_with_player_stats(self) -> set[str]:
        """Return the set of game IDs that already have player statistics stored."""

        with self.engine.begin() as conn:
            rows = conn.execute(select(self.player_stats.c.game_id).distinct()).fetchall()
        return {row[0] for row in rows}

    def latest_team_rating_week(self, season: str) -> Optional[int]:
        with self.engine.begin() as conn:
            row = conn.execute(
                select(func.max(self.team_unit_ratings.c.week)).where(self.team_unit_ratings.c.season == season)
            ).scalar()
        return row

    def record_backtest_metrics(
        self,
        run_id: str,
        model_name: str,
        metrics: Dict[str, float],
        sample_size: Optional[int] = None,
    ) -> None:
        if not metrics:
            return
        rows = [
            {
                "run_id": run_id,
                "model_name": model_name,
                "metric_name": metric,
                "metric_value": float(value),
                "sample_size": sample_size,
            }
            for metric, value in metrics.items()
        ]
        with self.engine.begin() as conn:
            conn.execute(self.model_backtests.insert(), rows)


# ---------------------------------------------------------------------------
# API clients
# ---------------------------------------------------------------------------


class MySportsFeedsClient:
    def __init__(self, user: str, password: str, timeout: int = 30):
        self.user = user
        self.password = password
        self.auth = (user, password)
        self.timeout = timeout

    def _request(self, endpoint: str, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{API_PREFIX_NFL}/{endpoint}"
        logging.debug("Requesting MySportsFeeds endpoint %s", url)
        resp = requests.get(url, params=params, auth=self.auth, timeout=self.timeout)
        resp.raise_for_status()
        try:
            return resp.json()
        except RequestsJSONDecodeError:
            content = resp.text.strip()
            if not content:
                logging.debug(
                    "Empty response body for MySportsFeeds endpoint %s; returning empty payload",
                    url,
                )
                return {}
            logging.warning(
                "Failed to decode JSON from MySportsFeeds endpoint %s (content-type=%s)",
                url,
                resp.headers.get("Content-Type"),
            )
            raise

    def fetch_games(self, season: str) -> List[Dict[str, Any]]:
        """Fetch the schedule for a season, retrying with alternative filters."""

        base_params: Dict[str, Any] = {"limit": 500}
        attempts: Tuple[Optional[str], ...] = (
            "completed,upcoming",
            "final,inprogress,scheduled",
            None,
        )

        for status_filter in attempts:
            params = dict(base_params)
            if status_filter:
                params["status"] = status_filter

            data = self._request(f"{season}/games.json", params=params)
            games = data.get("games", [])
            if games:
                if status_filter and status_filter != attempts[0]:
                    logging.debug(
                        "Fetched %d games for %s after retrying with status filter '%s'",
                        len(games),
                        season,
                        status_filter,
                    )
                return games

        logging.debug(
            "No games returned for %s even after retrying with multiple status filters",
            season,
        )
        return []

    def fetch_game_boxscore(self, season: str, game_id: str) -> Dict[str, Any]:
        return self._request(f"{season}/games/{game_id}/boxscore.json")

    def fetch_player_gamelogs(self, season: str, game_id: str) -> List[Dict[str, Any]]:
        try:
            data = self._request(
                f"{season}/games/{game_id}/player_gamelogs.json",
                params={"stats": "Rushing,Receiving,Passing,Fumbles"},
            )
        except HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 404:
                logging.debug(
                    "No player gamelogs found for season %s game %s (HTTP 404)",
                    season,
                    game_id,
                )
                return []
            raise
        return data.get("gamelogs", [])

    def fetch_injuries(
        self, season: Optional[str] = None, date: Optional[str] = None
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if season:
            params["season"] = season
        if date:
            params["date"] = date
        data = self._request("injuries.json", params=params or None)
        if not isinstance(data, dict):
            return {"players": [], "lastUpdatedOn": None}
        data.setdefault("players", [])
        return data

    def fetch_game_lineup(self, season: str, game_key: str) -> Dict[str, Any]:
        try:
            return self._request(f"{season}/games/{game_key}/lineup.json")
        except HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 404:
                logging.debug(
                    "No lineup available for season %s game %s (HTTP 404)",
                    season,
                    game_key,
                )
                return {}
            raise


class OddsApiClient:
    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout

    def _request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        url = f"{ODDS_BASE}/{endpoint}"
        params = params or {}
        params.update({"apiKey": self.api_key})
        logging.debug("Requesting Odds API endpoint %s", url)
        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        remaining = resp.headers.get("x-requests-remaining")
        if remaining is not None:
            logging.debug("Odds API requests remaining: %s", remaining)
        return resp.json()

    def fetch_odds(self) -> List[Dict[str, Any]]:
        params = {
            "regions": ",".join(ODDS_REGIONS),
            "oddsFormat": ODDS_FORMAT,
            "markets": "h2h",
        }
        return self._request(f"sports/{NFL_SPORT_KEY}/odds", params=params)


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MSFCreds:
    api_key: str
    password: str = "MYSPORTSFEEDS"


class NFLIngestor:
    def __init__(
        self,
        db: NFLDatabase,
        msf_client: MySportsFeedsClient,
        odds_client: OddsApiClient,
        supplemental_loader: SupplementalDataLoader,
    ):
        self.db = db
        self.msf_client = msf_client
        self.odds_client = odds_client
        self.supplemental_loader = supplemental_loader
        user = getattr(msf_client, "user", None)
        password = getattr(msf_client, "password", None)
        auth_tuple = getattr(msf_client, "auth", None)
        if not user and auth_tuple:
            try:
                user = auth_tuple[0]
            except Exception:
                user = None
        if not password and auth_tuple and len(auth_tuple) > 1:
            try:
                password = auth_tuple[1]
            except Exception:
                password = None
        self._msf_creds = MSFCreds(
            api_key=user or "",
            password=password or "MYSPORTSFEEDS",
        )

    def ingest(self, seasons: Iterable[str]) -> None:
        existing_games = self.db.fetch_existing_game_ids()
        games_with_stats = self.db.fetch_games_with_player_stats()
        logging.info("Found %d games already in database", len(existing_games))

        injuries_payload = self.msf_client.fetch_injuries()
        msf_injuries_last_updated = parse_dt(injuries_payload.get("lastUpdatedOn"))
        msf_injuries_by_team = self._group_msf_injuries(
            injuries_payload.get("players", []), msf_injuries_last_updated
        )
        lineup_cache: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}

        injury_rows_all: List[Dict[str, Any]] = []
        depth_rows_map: Dict[str, Dict[str, Any]] = {}
        advanced_rows_map: Dict[Tuple[str, int, str], Dict[str, Any]] = {}

        for season in seasons:
            games = self.msf_client.fetch_games(season)
            logging.info("Fetched %d games for season %s", len(games), season)
            if not games:
                logging.warning(
                    "No games returned from MySportsFeeds for %s. "
                    "Verify your API credentials, plan access, and season configuration.",
                    season,
                )

            new_game_rows: List[Dict[str, Any]] = []
            player_rows: List[Dict[str, Any]] = []

            for game in games:
                schedule = game.get("schedule", {})
                game_id = schedule.get("id")
                if not game_id:
                    continue

                game_id_str = str(game_id)
                have_player_stats = game_id_str in games_with_stats

                score = game.get("score") or {}
                home_score, away_score = self._extract_score_totals(score)
                start_time = parse_dt(schedule.get("startTime"))
                venue = schedule.get("venue") or {}
                weather = schedule.get("weather") or {}
                officials = schedule.get("officials") or []
                home_team_abbr = normalize_team_abbr(
                    (schedule.get("homeTeam") or {}).get("abbreviation")
                    or (schedule.get("homeTeam") or {}).get("name")
                )
                away_team_abbr = normalize_team_abbr(
                    (schedule.get("awayTeam") or {}).get("abbreviation")
                    or (schedule.get("awayTeam") or {}).get("name")
                )

                msf_injuries = self._collect_game_injuries(
                    game_id_str,
                    home_team_abbr,
                    away_team_abbr,
                    msf_injuries_by_team,
                )
                supplemental_injuries, _ = self.supplemental_loader.injuries_for_game(
                    game_id_str, home_team_abbr, away_team_abbr
                )
                injuries = self._merge_injury_rows(msf_injuries, supplemental_injuries)
                injury_summary = self._summarize_injury_rows(injuries)
                if injuries:
                    injury_rows_all.extend(injuries)

                for team_code in filter(None, {home_team_abbr, away_team_abbr}):
                    for depth_row in self.supplemental_loader.depth_chart_rows(team_code):
                        depth_rows_map[depth_row["depth_id"]] = depth_row

                lineup_rows = self._lineup_rows_from_msf(
                    start_time,
                    away_team_abbr,
                    home_team_abbr,
                    self._msf_creds,
                    lineup_cache,
                )
                try:
                    roster_rows = self._build_game_roster_rows(
                        game_id_str,
                        start_time,
                        home_team_abbr,
                        away_team_abbr,
                        lineup_rows,
                    )
                    if roster_rows:
                        self.db.upsert_game_rosters(roster_rows)
                        starters = sum(1 for row in roster_rows if row.get("is_starter") == 1)
                        logging.debug(
                            "Game %s roster upserted (rows=%d, starters=%d)",
                            game_id_str,
                            len(roster_rows),
                            starters,
                        )
                    else:
                        logging.info(
                            "No lineup roster rows for game %s (season=%s)",
                            game_id_str,
                            season,
                        )
                except Exception:
                    logging.exception(
                        "Failed building game roster rows for game %s",
                        game_id_str,
                    )
                for lineup_row in lineup_rows:
                    if lineup_row.get("game_start") is None:
                        lineup_row["game_start"] = start_time
                    depth_rows_map[lineup_row["depth_id"]] = lineup_row

                week_value = schedule.get("week")
                try:
                    week_int = int(week_value) if week_value is not None else None
                except (TypeError, ValueError):
                    week_int = None

                for team_code in filter(None, {home_team_abbr, away_team_abbr}):
                    advanced_payload = self.supplemental_loader.advanced_metrics(
                        season, week_int, team_code
                    )
                    if not advanced_payload:
                        continue
                    metric_id = advanced_payload.get("metric_id") or uuid.uuid4().hex
                    advanced_rows_map[(str(season), week_int or 0, team_code)] = {
                        "metric_id": metric_id,
                        "season": str(season),
                        "week": week_int,
                        "team": team_code,
                        "pace_seconds_per_play": self._safe_float(
                            advanced_payload.get("pace_seconds_per_play")
                            or advanced_payload.get("pace")
                        ),
                        "offense_epa": self._safe_float(advanced_payload.get("offense_epa")),
                        "defense_epa": self._safe_float(advanced_payload.get("defense_epa")),
                        "offense_success_rate": self._safe_float(
                            advanced_payload.get("offense_success_rate")
                        ),
                        "defense_success_rate": self._safe_float(
                            advanced_payload.get("defense_success_rate")
                        ),
                        "travel_penalty": self._safe_float(advanced_payload.get("travel_penalty")),
                        "rest_penalty": self._safe_float(advanced_payload.get("rest_penalty")),
                        "weather_adjustment": self._safe_float(
                            advanced_payload.get("weather_adjustment")
                        ),
                    }

                referee_name: Optional[str] = None
                if officials:
                    lead_official = officials[0] or {}
                    first = lead_official.get("firstName", "")
                    last = lead_official.get("lastName", "")
                    referee_name = f"{first} {last}".strip()
                    if not referee_name:
                        referee_name = lead_official.get("fullName")

                wind_mph = self._extract_wind_mph(weather.get("windSpeed"))
                humidity = self._extract_humidity(weather.get("humidity"))
                weather_override = self.supplemental_loader.weather_override(game_id_str)
                if weather_override:
                    if weather_override.get("temperature_f") is not None:
                        weather["temperature"] = weather_override.get("temperature_f")
                    if weather_override.get("conditions"):
                        weather["conditions"] = weather_override.get("conditions")
                    wind_mph = self._safe_float(weather_override.get("wind_mph") or wind_mph)
                    humidity = self._safe_float(weather_override.get("humidity") or humidity)
                    if weather_override.get("temperature_f") is not None:
                        weather_temperature = weather_override.get("temperature_f")
                    else:
                        weather_temperature = weather.get("temperature")
                else:
                    weather_temperature = weather.get("temperature")

                temperature_f = self._extract_temperature_fahrenheit(weather_temperature)
                wind_mph = self._safe_float(wind_mph)
                humidity = self._safe_float(humidity)

                new_game_rows.append(
                    {
                        "game_id": game_id_str,
                        "season": season,
                        "week": schedule.get("week"),
                        "start_time": start_time,
                        "venue": venue.get("name"),
                        "city": venue.get("city"),
                        "state": venue.get("state"),
                        "country": venue.get("country"),
                        "surface": venue.get("surface"),
                        "day_of_week": start_time.strftime("%A") if start_time else None,
                        "referee": referee_name,
                        "temperature_f": temperature_f,
                        "weather_conditions": weather.get("conditions"),
                        "wind_mph": wind_mph,
                        "humidity": humidity,
                        "injury_summary": injury_summary,
                        "home_team": home_team_abbr,
                        "away_team": away_team_abbr,
                        "home_score": home_score,
                        "away_score": away_score,
                        "status": schedule.get("status"),
                    }
                )

                status = (
                    schedule.get("status")
                    or schedule.get("playedStatus")
                    or (game.get("status") if isinstance(game, dict) else None)
                    or ""
                ).lower()
                is_completed = status.startswith("final") or status in {"completed", "postponed"}

                if have_player_stats:
                    logging.debug(
                        "Skipping player stats for already ingested game %s", game_id_str
                    )
                    continue

                if not is_completed:
                    logging.debug(
                        "Game %s in season %s has status '%s'; skipping player stats fetch until completion",
                        game_id_str,
                        season,
                        schedule.get("status"),
                    )
                    continue

                gamelog_entries = self.msf_client.fetch_player_gamelogs(season, game_id_str)
                player_entries = list(gamelog_entries)
                if not player_entries:
                    logging.debug(
                        "No player gamelog entries returned for season %s game %s", season, game_id_str
                    )
                    fallback_entries = self._fetch_boxscore_player_stats(season, game_id_str)
                    if fallback_entries:
                        logging.debug(
                            "Using boxscore fallback for season %s game %s player stats",
                            season,
                            game_id_str,
                        )
                        player_entries = fallback_entries
                if not player_entries:
                    continue
                for entry in player_entries:
                    player = entry.get("player", {})
                    team = entry.get("team", {})
                    stats = entry.get("stats", {})

                    def stat_value(stat_group: str, field: str) -> Optional[float]:
                        group = stats.get(stat_group, {})
                        value = group.get(field, {})
                        return value.get("#text") or value.get("value")

                    player_rows.append(
                        {
                            "game_id": game_id_str,
                            "player_id": str(player.get("id")),
                            "player_name": f"{player.get('firstName', '')} {player.get('lastName', '')}".strip(),
                            "team": team.get("abbreviation"),
                            "position": player.get("position"),
                            "rushing_attempts": self._safe_float(stat_value("Rushing", "RushingAttempts")),
                            "rushing_yards": self._safe_float(stat_value("Rushing", "RushingYards")),
                            "rushing_tds": self._safe_float(stat_value("Rushing", "RushingTD")),
                            "receiving_targets": self._safe_float(stat_value("Receiving", "Targets")),
                            "receptions": self._safe_float(stat_value("Receiving", "Receptions")),
                            "receiving_yards": self._safe_float(stat_value("Receiving", "ReceivingYards")),
                            "receiving_tds": self._safe_float(stat_value("Receiving", "ReceivingTD")),
                            "passing_attempts": self._safe_float(stat_value("Passing", "PassAttempts")),
                            "passing_completions": self._safe_float(stat_value("Passing", "PassCompletions")),
                            "passing_yards": self._safe_float(stat_value("Passing", "PassYards")),
                            "passing_tds": self._safe_float(stat_value("Passing", "PassTD")),
                            "fantasy_points": self._safe_float(stat_value("Fantasy", "FantasyPoints")),
                            "snap_count": self._safe_float(stat_value("Miscellaneous", "Snaps")),
                        }
                    )

            self.db.upsert_rows(self.db.games, new_game_rows, ["game_id"])
            self.db.upsert_rows(self.db.player_stats, player_rows, ["game_id", "player_id"])
            if len(new_game_rows) == 0 and len(player_rows) == 0:
                logging.warning(
                    "Ingested %d new games and %d player stat rows for %s. "
                    "If these counts are unexpectedly low, confirm that your MySportsFeeds subscription "
                    "includes detailed stats and that the targeted seasons contain completed games.",
                    len(new_game_rows),
                    len(player_rows),
                    season,
                )

        if injury_rows_all:
            self.db.upsert_rows(self.db.injury_reports, injury_rows_all, ["injury_id"])
        if depth_rows_map:
            self.db.upsert_rows(
                self.db.depth_charts,
                list(depth_rows_map.values()),
                ["depth_id"],
            )
        if advanced_rows_map:
            self.db.upsert_rows(
                self.db.team_advanced_metrics,
                list(advanced_rows_map.values()),
                ["metric_id"],
            )

        # Ingest odds separately as they change frequently (always upsert)
        self._ingest_odds()

    def _ingest_odds(self) -> None:
        odds_data = self.odds_client.fetch_odds()
        logging.info("Fetched %d odds entries", len(odds_data))

        odds_rows: List[Dict[str, Any]] = []
        for event in odds_data:
            commence_time = parse_dt(event.get("commence_time"))

            teams_list = [team for team in (event.get("teams") or []) if team]

            home_team_raw = event.get("home_team") or (teams_list[0] if teams_list else None)
            away_team_raw = event.get("away_team")
            if not away_team_raw and teams_list:
                away_team_raw = next(
                    (team for team in teams_list if team != home_team_raw),
                    teams_list[0] if teams_list else None,
                )

            home_team = normalize_team_abbr(home_team_raw)
            away_team = normalize_team_abbr(away_team_raw)

            if not home_team or not away_team:
                logging.debug(
                    "Skipping odds event %s due to unmapped team names (home=%s, away=%s)",
                    event.get("id"),
                    home_team_raw,
                    away_team_raw,
                )
                continue

            markets = event.get("bookmakers", [])
            if not markets:
                continue
            # Use the freshest bookmaker odds
            market = sorted(markets, key=lambda b: parse_dt(b.get("last_update")) or default_now_utc(), reverse=True)[0]
            last_update = parse_dt(market.get("last_update"))
            h2h = next((m for m in market.get("markets", []) if m.get("key") == "h2h"), None)
            if not h2h:
                continue

            prices = {outcome.get("name"): outcome.get("price") for outcome in h2h.get("outcomes", [])}
            home_price = prices.get(home_team)
            away_price = prices.get(away_team)

            def american_to_prob(odds: Optional[float]) -> Optional[float]:
                if odds is None:
                    return None
                odds = float(odds)
                if odds > 0:
                    return 100.0 / (odds + 100.0)
                return -odds / (-odds + 100.0)

            odds_rows.append(
                {
                    "game_id": event.get("id"),
                    "season": self._infer_season(commence_time),
                    "week": None,
                    "start_time": commence_time,
                    "venue": None,
                    "city": None,
                    "state": None,
                    "country": None,
                    "surface": None,
                    "day_of_week": commence_time.strftime("%A") if commence_time else None,
                    "referee": None,
                    "temperature_f": None,
                    "weather_conditions": None,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": None,
                    "away_score": None,
                    "status": "upcoming",
                    "home_moneyline": home_price,
                    "away_moneyline": away_price,
                    "home_implied_prob": american_to_prob(home_price),
                    "away_implied_prob": american_to_prob(away_price),
                    "odds_updated": last_update,
                }
            )

        self.db.upsert_rows(
            self.db.games,
            odds_rows,
            ["game_id"],
            update_columns=[
                "start_time",
                "home_moneyline",
                "away_moneyline",
                "home_implied_prob",
                "away_implied_prob",
                "odds_updated",
                "home_team",
                "away_team",
            ],
        )

    def _group_msf_injuries(
        self,
        players: List[Dict[str, Any]],
        last_updated: Optional[dt.datetime],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for entry in players or []:
            team_info = entry.get("currentTeam") or entry.get("team") or {}
            team_abbr = normalize_team_abbr(
                team_info.get("abbreviation") or team_info.get("name")
            )
            if not team_abbr:
                continue

            injury_info = entry.get("currentInjury") or {}
            roster_status = entry.get("currentRosterStatus") or entry.get("rosterStatus")
            roster_status_text = str(roster_status or "").strip()
            roster_status_normalized = roster_status_text.lower()
            if not injury_info:
                if roster_status_normalized and (
                    re.search(r"\b(ir|injured|reserve|pup|nfi)\b", roster_status_normalized)
                    or any(keyword in roster_status_normalized for keyword in INJURY_OUT_KEYWORDS)
                ):
                    injury_info = {
                        "status": roster_status_text,
                        "playingProbability": roster_status_text,
                        "description": roster_status_text,
                    }
            if not injury_info:
                continue

            first = entry.get("firstName", "")
            last = entry.get("lastName", "")
            player_name = " ".join(part for part in [first, last] if part).strip()
            if not player_name:
                player_name = entry.get("displayName") or ""
            if not player_name:
                continue

            position = normalize_position(
                entry.get("primaryPosition")
                or entry.get("position")
                or (injury_info.get("position") if isinstance(injury_info, dict) else None)
            )

            status = injury_info.get("playingProbability") or injury_info.get("status")
            practice_status = (
                injury_info.get("practiceStatus")
                or injury_info.get("practice")
                or entry.get("currentPracticeStatus")
                or roster_status_text
            )
            description = injury_info.get("description")

            reported_at: Optional[dt.datetime]
            updated_raw = injury_info.get("updatedOn") if isinstance(injury_info, dict) else None
            if updated_raw:
                reported_at = parse_dt(updated_raw)
            else:
                reported_at = last_updated

            grouped.setdefault(team_abbr, []).append(
                {
                    "injury_id": f"msf-{entry.get('id') or uuid.uuid4().hex}",
                    "team": team_abbr,
                    "player_name": player_name,
                    "status": status,
                    "practice_status": practice_status,
                    "description": description,
                    "report_time": reported_at,
                    "position": position,
                }
            )

        return grouped

    def _collect_game_injuries(
        self,
        game_id: str,
        home_team: Optional[str],
        away_team: Optional[str],
        grouped: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for team_code in filter(None, {normalize_team_abbr(home_team), normalize_team_abbr(away_team)}):
            for base_row in grouped.get(team_code, []):
                player_key = normalize_player_name(base_row.get("player_name"))
                if not player_key:
                    continue
                row = dict(base_row)
                row["team"] = team_code
                row["game_id"] = str(game_id)
                base_id = row.get("injury_id") or uuid.uuid4().hex
                row["injury_id"] = f"{game_id}:{base_id}"
                if row.get("report_time") and not isinstance(row["report_time"], dt.datetime):
                    row["report_time"] = parse_dt(row["report_time"])
                row["position"] = normalize_position(row.get("position"))
                rows.append(row)
        return rows

    def _merge_injury_rows(
        self,
        msf_rows: List[Dict[str, Any]],
        supplemental_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: Dict[Tuple[str, str], Dict[str, Any]] = {}

        def _ingest(rows: List[Dict[str, Any]], priority: int) -> None:
            for row in rows or []:
                team = normalize_team_abbr(row.get("team"))
                player_name = row.get("player_name")
                player_key = normalize_player_name(player_name)
                if not team or not player_key:
                    continue

                record = dict(row)
                record["team"] = team
                record["player_name"] = player_name
                record["position"] = normalize_position(record.get("position"))
                record.setdefault("game_id", row.get("game_id"))
                record.setdefault("injury_id", uuid.uuid4().hex)
                if record.get("injury_id") and record.get("game_id"):
                    if not str(record["injury_id"]).startswith(str(record["game_id"])):
                        record["injury_id"] = f"{record['game_id']}:{record['injury_id']}"

                if record.get("report_time") and not isinstance(record["report_time"], dt.datetime):
                    record["report_time"] = parse_dt(record["report_time"])

                key = (team, player_key)
                existing = merged.get(key)
                if not existing:
                    record["_priority"] = priority
                    merged[key] = record
                    continue

                # prefer lower priority value (0 beats 1) but always merge fresh details
                if existing.get("_priority", priority) > priority:
                    for field in ("status", "practice_status", "description", "position", "report_time"):
                        if not record.get(field) and existing.get(field):
                            record[field] = existing[field]
                    record["_priority"] = priority
                    merged[key] = record
                else:
                    for field in ("status", "practice_status", "description", "position"):
                        if not existing.get(field) and record.get(field):
                            existing[field] = record[field]
                    if record.get("report_time") and (
                        not existing.get("report_time")
                        or (
                            isinstance(existing.get("report_time"), dt.datetime)
                            and isinstance(record.get("report_time"), dt.datetime)
                            and record["report_time"] > existing["report_time"]
                        )
                    ):
                        existing["report_time"] = record["report_time"]

        _ingest(msf_rows, 0)
        _ingest(supplemental_rows, 1)

        output: List[Dict[str, Any]] = []
        for record in merged.values():
            record.pop("_priority", None)
            game_id = record.get("game_id")
            if not game_id:
                continue
            bucket = compute_injury_bucket(record.get("status"), record.get("description"))
            practice_bucket = normalize_practice_status(record.get("practice_status"))
            if bucket == "other" and practice_bucket == "available":
                continue
            record["practice_status"] = record.get("practice_status")
            record["status"] = record.get("status")
            output.append(record)
        return output

    def _summarize_injury_rows(self, injuries: List[Dict[str, Any]]) -> Optional[str]:
        if not injuries:
            return None
        parts: List[str] = []
        for row in injuries:
            player_name = row.get("player_name")
            if not player_name:
                continue
            bucket = compute_injury_bucket(row.get("status"), row.get("description"))
            practice_bucket = normalize_practice_status(row.get("practice_status"))
            if bucket == "other" and practice_bucket == "available":
                continue
            label = bucket if bucket != "other" else practice_bucket
            if not label:
                continue
            parts.append(f"{player_name}({label})")
        if not parts:
            return None
        preview = parts[:6]
        summary = ", ".join(preview)
        remaining = len(parts) - len(preview)
        if remaining > 0:
            summary += f" +{remaining} more"
        return summary

    def _lineup_rows_from_msf(
        self,
        start_time_utc: Optional[dt.datetime],
        away_team_abbr: Optional[str],
        home_team_abbr: Optional[str],
        msf_creds: MSFCreds,
        lineup_cache: Dict[Tuple[str, str, str], List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if isinstance(start_time_utc, str):
            start_dt = parse_dt(start_time_utc)
        else:
            start_dt = start_time_utc
        if not start_dt or not away_team_abbr or not home_team_abbr:
            logging.info(
                "lineup: missing keys start=%s away=%s home=%s",
                start_time_utc,
                away_team_abbr,
                home_team_abbr,
            )
            return []

        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=dt.timezone.utc)

        away_norm = _msf_team_abbr(away_team_abbr)
        home_norm = _msf_team_abbr(home_team_abbr)
        if not away_norm or not home_norm:
            logging.info(
                "lineup: could not normalize abbreviations away=%s home=%s",
                away_team_abbr,
                home_team_abbr,
            )
            return []

        local_date = _home_local_game_date(start_dt, home_norm)
        utc_date = start_dt.astimezone(dt.timezone.utc).date()

        season_slug = _nfl_season_slug_for_start(start_dt)
        if not season_slug:
            logging.info("lineup: unable to determine season slug for %s", start_dt)
            return []

        date_candidates: List[str] = []
        local_key = f"{local_date.year:04d}{local_date.month:02d}{local_date.day:02d}"
        date_candidates.append(local_key)
        utc_key = f"{utc_date.year:04d}{utc_date.month:02d}{utc_date.day:02d}"
        if utc_key not in date_candidates:
            date_candidates.append(utc_key)

        if not msf_creds or not msf_creds.api_key:
            logging.warning("lineup: missing MySportsFeeds credentials; skipping fetch")
            return []
        auth = HTTPBasicAuth(msf_creds.api_key, msf_creds.password or "MYSPORTSFEEDS")
        accept_headers = {"Accept": "application/json"}

        last_payload: Optional[Dict[str, Any]] = None

        for date_key in date_candidates:
            for lineup_type in (None, "expected"):
                cache_token = f"{date_key}|{lineup_type or 'default'}"
                cache_key = (cache_token, away_norm, home_norm)
                if cache_key in lineup_cache:
                    cached = lineup_cache[cache_key]
                    if cached:
                        return cached
                    continue

                url = (
                    f"https://api.mysportsfeeds.com/v2.1/pull/nfl/{season_slug}/games/"
                    f"{date_key}-{away_norm}-{home_norm}/lineup.json"
                )

                params = {"lineupType": lineup_type} if lineup_type else None
                response = _http_get_with_retry(
                    url,
                    auth,
                    params=params,
                    headers=accept_headers,
                )

                if response is None:
                    logging.info("lineup: HTTP failed for %s", url)
                    lineup_cache[cache_key] = []
                    continue

                if response.status_code == 404:
                    logging.info(
                        "lineup: 404 not found for %s (season slug or date/abbr mismatch)",
                        url,
                    )
                    lineup_cache[cache_key] = []
                    continue

                if response.status_code == 401:
                    logging.warning(
                        "lineup: 401 unauthorized for %s (check MSF credentials)",
                        url,
                    )
                    lineup_cache[cache_key] = []
                    return []

                if response.status_code == 204:
                    logging.info(
                        "lineup: 204 empty response for %s (type=%s)",
                        url,
                        lineup_type or "actual",
                    )
                    lineup_cache[cache_key] = []
                    continue

                if response.status_code != 200:
                    logging.warning(
                        "lineup: %s returned %s",
                        url,
                        response.status_code,
                    )
                    lineup_cache[cache_key] = []
                    continue

                try:
                    payload = response.json()
                except Exception:
                    logging.exception("lineup: JSON decode failed for %s", url)
                    lineup_cache[cache_key] = []
                    continue

                last_payload = payload if isinstance(payload, dict) else None
                last_updated = (
                    parse_dt(payload.get("lastUpdatedOn"))
                    if isinstance(payload, dict)
                    else None
                )
                rows = _extract_lineup_rows(payload if isinstance(payload, dict) else {})

                if not rows:
                    team_blocks = (
                        payload.get("teamLineups")
                        if isinstance(payload, dict)
                        else None
                    ) or []
                    details: List[str] = []
                    for block in team_blocks:
                        team_label = (block.get("team") or {}).get("abbreviation")
                        actual = (block.get("actual") or {}).get("lineupPositions") or []
                        expected = (block.get("expected") or {}).get("lineupPositions") or []
                        details.append(
                            f"{team_label}: actual={len(actual)} expected={len(expected)}"
                        )
                    logging.info(
                        "lineup: empty lineupPositions for %s; %s",
                        url,
                        "; ".join(details) if details else "no teamLineups",
                    )

                enriched_rows: List[Dict[str, Any]] = []
                for record in rows:
                    team = record.get("team")
                    position = record.get("position")
                    if not team or not position:
                        continue
                    player_name = record.get("player_name") or ""
                    player_id = record.get("player_id") or ""
                    first_name = record.get("first_name") or ""
                    last_name = record.get("last_name") or ""
                    name_for_key = " ".join(part for part in [first_name, last_name] if part) or player_name
                    player_key = record.get("__pname_key") or robust_player_name_key(name_for_key)
                    if not player_key:
                        continue
                    depth_id = (
                        f"msf-lineup:{team}:{position}:{player_id}"
                        if player_id
                        else f"msf-lineup:{team}:{position}:{player_key}"
                    )
                enriched_rows.append(
                    {
                        "team": team,
                        "position": position,
                        "player_id": player_id,
                        "player_name": player_name,
                        "first_name": first_name,
                        "last_name": last_name,
                        "rank": record.get("rank"),
                        "depth_id": depth_id,
                        "updated_at": last_updated,
                        "source": "msf-lineup",
                        "player_team": record.get("player_team"),
                        "game_start": start_dt,
                        "__pname_key": player_key,
                        "side": record.get("side"),
                        "base_pos": record.get("base_pos") or position,
                        "playing_probability": record.get("playing_probability"),
                        "status_bucket": record.get("status_bucket"),
                        "practice_status": record.get("practice_status"),
                    }
                )

                lineup_cache[cache_key] = enriched_rows
                if enriched_rows:
                    return enriched_rows

        if last_payload is not None:
            logging.debug(
                "lineup: no usable rows after attempts for %s @ %s (payload timestamp=%s)",
                away_norm,
                home_norm,
                last_payload.get("lastUpdatedOn") if isinstance(last_payload, dict) else None,
            )

        return []

    def _skill_pos(self, pos: str) -> bool:
        return normalize_position(pos) in {"QB", "RB", "WR", "TE"}

    def _is_starter_label(self, base_pos: str, rank: Optional[int]) -> bool:
        if base_pos == "QB":
            return (rank or 99) == 1
        if base_pos == "RB":
            return (rank or 99) == 1
        if base_pos == "WR":
            return (rank or 99) in {1, 2, 3}
        if base_pos == "TE":
            return (rank or 99) == 1
        return False

    def _build_game_roster_rows(
        self,
        game_id: str,
        start_time: Optional[dt.datetime],
        home_team_abbr: Optional[str],
        away_team_abbr: Optional[str],
        lineup_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not lineup_rows:
            return []

        rows: List[Dict[str, Any]] = []
        lineup_by_team: Dict[str, List[Dict[str, Any]]] = {}
        for record in lineup_rows:
            team_code = normalize_team_abbr(record.get("team"))
            if not team_code:
                continue
            lineup_by_team.setdefault(team_code, []).append(record)

        for team in filter(None, {home_team_abbr, away_team_abbr}):
            team_normalized = normalize_team_abbr(team)
            if not team_normalized:
                continue
            team_entries = lineup_by_team.get(team_normalized, [])
            if not team_entries:
                continue

            by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for entry in team_entries:
                pos = normalize_position(entry.get("position"))
                if not self._skill_pos(pos):
                    continue
                lineup_player_team = normalize_team_abbr(entry.get("player_team"))
                if pos in {"QB", "RB", "WR", "TE"}:
                    if lineup_player_team and lineup_player_team != team_normalized:
                        continue
                pid = (entry.get("player_id") or "").strip()
                pname = (entry.get("player_name") or "").strip()
                if not pname and not pid:
                    continue
                key = (pid or "", pname)
                rank_val = entry.get("rank")
                parsed_rank: Optional[int]
                if isinstance(rank_val, (int, float)) and not math.isnan(rank_val):
                    parsed_rank = int(rank_val)
                else:
                    parsed_rank = None
                status_bucket = entry.get("status_bucket") or "other"
                practice_status = normalize_practice_status(
                    entry.get("practice_status")
                )
                current = by_key.get(key)
                current_rank = current.get("rank") if current else None
                current_status = current.get("status_bucket") if current else None
                new_rank_val = parsed_rank or 999
                current_rank_val = current_rank or 999
                replace_entry = False
                if current is None:
                    replace_entry = True
                elif new_rank_val < current_rank_val:
                    replace_entry = True
                elif new_rank_val == current_rank_val:
                    current_inactive = current_status in INACTIVE_INJURY_BUCKETS if current_status else False
                    new_inactive = status_bucket in INACTIVE_INJURY_BUCKETS
                    if current_inactive and not new_inactive:
                        replace_entry = True
                    elif not current_inactive and new_inactive:
                        replace_entry = False
                    else:
                        existing_ts = current.get("updated_at")
                        new_ts = entry.get("updated_at")
                        existing_dt = (
                            existing_ts
                            if isinstance(existing_ts, dt.datetime)
                            else parse_dt(existing_ts)
                        )
                        new_dt = (
                            new_ts if isinstance(new_ts, dt.datetime) else parse_dt(new_ts)
                        )
                        if new_dt and (not existing_dt or new_dt > existing_dt):
                            replace_entry = True

                if replace_entry:
                    by_key[key] = {
                        "player_id": pid,
                        "player_name": pname,
                        "position": pos,
                        "rank": parsed_rank,
                        "source": entry.get("source", "msf-lineup"),
                        "updated_at": entry.get("updated_at"),
                        "status_bucket": status_bucket,
                        "practice_status": practice_status,
                    }

            now_utc = default_now_utc()
            for info in by_key.values():
                updated_at = info.get("updated_at")
                if isinstance(updated_at, str):
                    updated_at_dt = parse_dt(updated_at)
                elif isinstance(updated_at, dt.datetime):
                    updated_at_dt = updated_at
                else:
                    updated_at_dt = None
                if updated_at_dt is None:
                    updated_at_dt = now_utc
                rows.append(
                    {
                        "game_id": str(game_id),
                        "team": team_normalized,
                        "player_id": info["player_id"],
                        "player_name": info["player_name"],
                        "position": info["position"],
                        "depth_rank": info["rank"],
                        "is_starter": 1
                        if (
                            self._is_starter_label(info["position"], info["rank"])
                            and info.get("status_bucket") not in INACTIVE_INJURY_BUCKETS
                        )
                        else 0,
                        "source": info.get("source", "msf-lineup"),
                        "updated_at": updated_at_dt,
                        "game_start": start_time,
                    }
                )
        return rows

    def fetch_lineup_rows(
        self,
        start_time: Optional[Union[dt.datetime, str]],
        away_team: Optional[str],
        home_team: Optional[str],
        cache: Optional[Dict[Tuple[str, str, str], List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """Public helper to retrieve MSF lineup rows for the given matchup."""

        cache = cache or {}
        return self._lineup_rows_from_msf(
            start_time,
            away_team,
            home_team,
            self._msf_creds,
            cache,
        )

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_temperature_fahrenheit(value: Any) -> Optional[float]:
        """Normalize temperature payloads into a Fahrenheit float."""

        if value is None:
            return None

        if isinstance(value, dict):
            candidates = [
                value.get("fahrenheit"),
                value.get("F"),
                value.get("tempF"),
                value.get("value"),
            ]
            for candidate in candidates:
                result = NFLIngestor._safe_float(candidate)
                if result is not None:
                    return result
            return None

        if isinstance(value, (list, tuple)):
            for item in value:
                result = NFLIngestor._extract_temperature_fahrenheit(item)
                if result is not None:
                    return result
            return None

        return NFLIngestor._safe_float(value)

    @staticmethod
    def _infer_season(start_time: Optional[dt.datetime]) -> Optional[str]:
        if not start_time:
            return None
        year = start_time.year
        if start_time.month < 3:
            year -= 1
        return f"{year}-regular"

    @staticmethod
    def _extract_wind_mph(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, dict):
            for key in ("milesPerHour", "mph", "value", "speed", "#text"):
                if key in value:
                    result = NFLIngestor._safe_float(value[key])
                    if result is not None:
                        return result
        return NFLIngestor._safe_float(value)

    @staticmethod
    def _extract_humidity(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, dict):
            for key in ("percent", "humidity", "value", "#text"):
                if key in value:
                    result = NFLIngestor._safe_float(value[key])
                    if result is not None:
                        return result
        return NFLIngestor._safe_float(value)

    def _fetch_boxscore_player_stats(self, season: str, game_id: str) -> List[Dict[str, Any]]:
        """Fallback to boxscore endpoint when detailed gamelogs are unavailable."""

        try:
            boxscore = self.msf_client.fetch_game_boxscore(season, game_id)
        except HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status == 404:
                logging.debug(
                    "Boxscore not available for season %s game %s (HTTP 404)",
                    season,
                    game_id,
                )
                return []
            logging.debug(
                "Failed to fetch boxscore for season %s game %s: %s",
                season,
                game_id,
                exc,
            )
            return []

        game_info = boxscore.get("game", {}) or {}
        team_lookup = {
            "home": (game_info.get("homeTeam") or {}).get("abbreviation"),
            "away": (game_info.get("awayTeam") or {}).get("abbreviation"),
        }

        stats_root = boxscore.get("stats") or {}
        normalized: List[Dict[str, Any]] = []
        for side in ("home", "away"):
            side_payload = stats_root.get(side) or {}
            players = side_payload.get("players") or []
            team_abbr = team_lookup.get(side)
            for player_entry in players:
                if not isinstance(player_entry, dict):
                    continue
                player_stats = self._normalize_boxscore_stat_groups(player_entry.get("playerStats"))
                normalized.append(
                    {
                        "player": player_entry.get("player", {}) or {},
                        "team": {"abbreviation": team_abbr} if team_abbr else {},
                        "stats": player_stats,
                    }
                )

        return normalized

    @staticmethod
    def _normalize_boxscore_stat_groups(raw_groups: Any) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Convert boxscore player stat groups to the gamelog-style schema."""

        if not isinstance(raw_groups, list):
            return {}

        normalized: Dict[str, Dict[str, Dict[str, Any]]] = {}

        def assign(group: str, target: str, value: Any) -> None:
            if value in (None, ""):
                return
            normalized.setdefault(group, {})[target] = {"value": value}

        for group_entry in raw_groups:
            if not isinstance(group_entry, dict):
                continue
            for group_name, metrics in group_entry.items():
                if not isinstance(metrics, dict):
                    continue
                key = group_name.lower()
                if key == "rushing":
                    assign("Rushing", "RushingAttempts", metrics.get("rushAttempts"))
                    assign("Rushing", "RushingYards", metrics.get("rushYards"))
                    assign("Rushing", "RushingTD", metrics.get("rushTD"))
                elif key == "receiving":
                    assign("Receiving", "Targets", metrics.get("targets"))
                    assign("Receiving", "Receptions", metrics.get("receptions"))
                    assign("Receiving", "ReceivingYards", metrics.get("recYards"))
                    assign("Receiving", "ReceivingTD", metrics.get("recTD"))
                elif key == "passing":
                    assign("Passing", "PassAttempts", metrics.get("passAttempts"))
                    assign("Passing", "PassCompletions", metrics.get("passCompletions"))
                    assign("Passing", "PassYards", metrics.get("passYards"))
                    assign("Passing", "PassTD", metrics.get("passTD"))
                elif key == "fumbles":
                    assign("Fumbles", "Fumbles", metrics.get("fumbles"))
                elif key == "snapcounts":
                    offense_snaps = metrics.get("offenseSnaps")
                    if offense_snaps is not None:
                        assign("Miscellaneous", "Snaps", offense_snaps)

        return normalized

    @staticmethod
    def _extract_score_totals(score_payload: Any) -> Tuple[Optional[float], Optional[float]]:
        """Extract final home and away scores from the flexible MSF schedule payload."""

        if not isinstance(score_payload, dict):
            return None, None

        def first_numeric(mapping: Dict[str, Any], candidates: Tuple[str, ...]) -> Optional[float]:
            for key in candidates:
                if key not in mapping or mapping[key] in (None, ""):
                    continue
                value = mapping[key]
                if isinstance(value, dict):
                    for inner_key in ("#text", "value", "total", "score", "amount"):
                        inner_val = value.get(inner_key)
                        parsed = NFLIngestor._safe_float(inner_val)
                        if parsed is not None:
                            return parsed
                    parsed = NFLIngestor._safe_float(value)
                    if parsed is not None:
                        return parsed
                else:
                    parsed = NFLIngestor._safe_float(value)
                    if parsed is not None:
                        return parsed
            return None

        home_candidates = (
            "homeScore",
            "homeScoreTotal",
            "homeScoreFinal",
            "homePoints",
            "homeScoreValue",
        )
        away_candidates = (
            "awayScore",
            "awayScoreTotal",
            "awayScoreFinal",
            "awayPoints",
            "awayScoreValue",
        )

        return first_numeric(score_payload, home_candidates), first_numeric(score_payload, away_candidates)


# ---------------------------------------------------------------------------
# Feature engineering & modeling
# ---------------------------------------------------------------------------


class FeatureBuilder:
    """Transforms raw database tables into model-ready feature sets."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.games_frame: Optional[pd.DataFrame] = None
        self.player_feature_frame: Optional[pd.DataFrame] = None
        self.team_strength_frame: Optional[pd.DataFrame] = None
        self.team_strength_latest_by_season: Optional[pd.DataFrame] = None
        self.team_strength_latest_overall: Optional[pd.DataFrame] = None
        self.team_history_frame: Optional[pd.DataFrame] = None
        self.team_history_latest_by_season: Optional[pd.DataFrame] = None
        self.team_history_latest_overall: Optional[pd.DataFrame] = None
        self.context_feature_frame: Optional[pd.DataFrame] = None
        self.injury_frame: Optional[pd.DataFrame] = None
        self.depth_chart_frame: Optional[pd.DataFrame] = None
        self.advanced_metrics_frame: Optional[pd.DataFrame] = None

    def load_dataframes(
        self,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        games = pd.read_sql_table("nfl_games", self.engine)
        player_stats = pd.read_sql_table("nfl_player_stats", self.engine)
        team_ratings = pd.read_sql_table("nfl_team_unit_ratings", self.engine)
        injuries = pd.read_sql_table("nfl_injury_reports", self.engine)
        depth_charts = pd.read_sql_table("nfl_depth_charts", self.engine)
        advanced_metrics = pd.read_sql_table("nfl_team_advanced_metrics", self.engine)

        # Normalize column names to plain strings so downstream pipelines see
        # consistent labels regardless of database dialect.
        games = games.rename(columns=lambda col: str(col))
        player_stats = player_stats.rename(columns=lambda col: str(col))
        team_ratings = team_ratings.rename(columns=lambda col: str(col))
        injuries = injuries.rename(columns=lambda col: str(col))
        depth_charts = depth_charts.rename(columns=lambda col: str(col))
        advanced_metrics = advanced_metrics.rename(columns=lambda col: str(col))
        if "position" in player_stats.columns:
            player_stats["position"] = player_stats["position"].apply(normalize_position)
        if "practice_status" in player_stats.columns:
            player_stats["practice_status"] = player_stats["practice_status"].apply(
                normalize_practice_status
            )
        if "position" in depth_charts.columns:
            depth_charts["position"] = depth_charts["position"].apply(normalize_position)
        if "practice_status" in injuries.columns:
            injuries["practice_status_raw"] = injuries["practice_status"]
            injuries["practice_status"] = injuries["practice_status"].apply(
                normalize_practice_status
            )
        if "status" in injuries.columns:
            injuries["status_original"] = injuries["status"]
            injuries["status"] = injuries["status"].apply(normalize_injury_status)

        return games, player_stats, team_ratings, injuries, depth_charts, advanced_metrics

    def build_features(self) -> Dict[str, pd.DataFrame]:
        (
            games,
            player_stats,
            team_ratings,
            injuries,
            depth_charts,
            advanced_metrics,
        ) = self.load_dataframes()

        if games.empty:
            logging.warning("No games available in the database. Skipping model training.")
            return {}

        games = games.copy()
        player_stats = player_stats.copy()
        injuries = injuries.copy()
        depth_charts = depth_charts.copy()
        advanced_metrics = advanced_metrics.copy()

        if "position" in player_stats.columns:
            player_stats["position"] = player_stats["position"].apply(normalize_position)
        self.games_frame = games
        self.injury_frame = injuries
        self.depth_chart_frame = depth_charts
        self.advanced_metrics_frame = advanced_metrics

        # Basic cleanup
        games["start_time"] = pd.to_datetime(games["start_time"])
        games["day_of_week"] = games["day_of_week"].fillna(
            games["start_time"].dt.day_name()
        )
        games["game_result"] = np.where(
            games["home_score"] > games["away_score"], "home",
            np.where(games["home_score"] < games["away_score"], "away", "push"),
        )

        if not injuries.empty:
            injuries["team"] = injuries["team"].apply(normalize_team_abbr)
            injuries["player_name_norm"] = injuries["player_name"].apply(normalize_player_name)
            if "status_original" not in injuries.columns:
                injuries["status_original"] = injuries.get("status")
            injuries["status_bucket"] = injuries.apply(
                lambda row: compute_injury_bucket(
                    row.get("status_original") or row.get("status"),
                    row.get("description"),
                ),
                axis=1,
            )
            injuries["practice_status"] = injuries["practice_status"].fillna("")
            injuries["report_time"] = pd.to_datetime(injuries["report_time"], errors="coerce")
            injury_counts = (
                injuries.groupby(["game_id", "team", "status_bucket"]).size().unstack(fill_value=0).reset_index()
            )
            value_columns = [col for col in injury_counts.columns if col not in {"game_id", "team"}]
            injury_counts["injury_total"] = injury_counts[value_columns].sum(axis=1)

            home_injuries = injury_counts.rename(columns={"team": "home_team"})
            home_injuries = home_injuries.rename(
                columns={col: f"home_injury_{col}" for col in home_injuries.columns if col not in {"game_id", "home_team"}}
            )
            games = games.merge(home_injuries, on=["game_id", "home_team"], how="left")

            away_injuries = injury_counts.rename(columns={"team": "away_team"})
            away_injuries = away_injuries.rename(
                columns={col: f"away_injury_{col}" for col in away_injuries.columns if col not in {"game_id", "away_team"}}
            )
            games = games.merge(away_injuries, on=["game_id", "away_team"], how="left")

            injury_feature_columns = [col for col in games.columns if col.endswith("_injury_total") or "_injury_" in col]
            for col in injury_feature_columns:
                games[col] = games[col].fillna(0.0)

        # Derive rolling scoring, rest, and win-rate indicators from historical games.
        team_game_history = self._compute_team_game_rolling_stats(games)
        self.team_history_frame = team_game_history
        penalties_by_week = pd.DataFrame()
        if not team_game_history.empty:
            penalties_by_week = (
                team_game_history.groupby(["season", "week", "team"], as_index=False)[
                    ["travel_penalty", "rest_penalty", "timezone_diff_hours"]
                ]
                .mean()
                .rename(columns={"timezone_diff_hours": "avg_timezone_diff_hours"})
            )
        if not team_game_history.empty:
            history_sorted = team_game_history.sort_values(["team", "season", "start_time"])
            self.team_history_latest_by_season = history_sorted.drop_duplicates(
                subset=["team", "season"], keep="last"
            )
            self.team_history_latest_overall = history_sorted.drop_duplicates(
                subset=["team"], keep="last"
            )
        else:
            empty_history = team_game_history.iloc[0:0]
            self.team_history_latest_by_season = empty_history
            self.team_history_latest_overall = empty_history

        datasets: Dict[str, pd.DataFrame] = {}
        team_strength: pd.DataFrame

        def _merge_penalties_into_strength(strength: pd.DataFrame) -> pd.DataFrame:
            if strength is None or strength.empty or penalties_by_week.empty:
                return strength
            merged_strength = strength.merge(
                penalties_by_week,
                on=["season", "week", "team"],
                how="left",
                suffixes=("", "_hist"),
            )
            for col in ["travel_penalty", "rest_penalty", "avg_timezone_diff_hours"]:
                hist_col = f"{col}_hist"
                if hist_col in merged_strength:
                    merged_strength[col] = merged_strength[col].combine_first(
                        merged_strength[hist_col]
                    )
                    merged_strength.drop(columns=[hist_col], inplace=True)
            return merged_strength

        if player_stats.empty:
            logging.warning(
                "Player statistics table is empty. Player-level models will not be trained."
            )
            team_strength = self._compute_team_unit_strength(player_stats, advanced_metrics)
            team_strength = _merge_penalties_into_strength(team_strength)
            self.team_strength_frame = team_strength
        else:
            enrichment_columns = [
                "game_id",
                "season",
                "week",
                "start_time",
                "venue",
                "city",
                "state",
                "day_of_week",
                "referee",
                "weather_conditions",
                "temperature_f",
                "home_team",
                "away_team",
            ]
            player_stats = player_stats.merge(
                games[enrichment_columns],
                on="game_id",
                how="left",
            )

            team_strength = self._compute_team_unit_strength(player_stats, advanced_metrics)
            team_strength = _merge_penalties_into_strength(team_strength)

            player_stats = player_stats.merge(
                team_strength,
                on=["team", "season", "week"],
                how="left",
                suffixes=("", "_team"),
            )

            opponent_strength = team_strength.rename(
                columns={
                    "team": "opponent",
                    "offense_pass_rating": "opp_offense_pass_rating",
                    "offense_rush_rating": "opp_offense_rush_rating",
                    "defense_pass_rating": "opp_defense_pass_rating",
                    "defense_rush_rating": "opp_defense_rush_rating",
                    "pace_seconds_per_play": "opp_pace_seconds_per_play",
                    "offense_epa": "opp_offense_epa",
                    "defense_epa": "opp_defense_epa",
                    "offense_success_rate": "opp_offense_success_rate",
                    "defense_success_rate": "opp_defense_success_rate",
                    "travel_penalty": "opp_travel_penalty",
                    "rest_penalty": "opp_rest_penalty",
                    "weather_adjustment": "opp_weather_adjustment",
                    "avg_timezone_diff_hours": "opp_timezone_diff_hours",
                }
            )

            player_stats = player_stats.merge(
                games[["game_id", "home_team", "away_team"]],
                on="game_id",
                how="left",
                suffixes=("", "_game"),
            )

            player_stats["opponent"] = np.where(
                player_stats["team"] == player_stats["home_team"],
                player_stats["away_team"],
                player_stats["home_team"],
            )

            player_stats = player_stats.merge(
                opponent_strength,
                on=["opponent", "season", "week"],
                how="left",
            )

            context_features = self._compute_contextual_averages(player_stats)
            self.context_feature_frame = context_features
            player_stats = player_stats.merge(
                context_features,
                on=["team", "venue", "day_of_week", "referee"],
                how="left",
            )

            player_stats["player_name_norm"] = player_stats["player_name"].apply(
                normalize_player_name
            )

            if not injuries.empty:
                injuries_latest = injuries.copy()
                if "player_name_norm" not in injuries_latest.columns:
                    injuries_latest["player_name_norm"] = injuries_latest["player_name"].apply(
                        normalize_player_name
                    )
                injuries_latest = injuries_latest[injuries_latest["player_name_norm"] != ""]
                injuries_latest = injuries_latest.sort_values("report_time")
                injuries_latest = injuries_latest.drop_duplicates(
                    subset=["game_id", "team", "player_name_norm"], keep="last"
                )
                injuries_subset = injuries_latest[
                    [
                        "game_id",
                        "team",
                        "player_name_norm",
                        "status_bucket",
                        "status",
                        "practice_status",
                    ]
                ]
                player_stats = player_stats.merge(
                    injuries_subset,
                    on=["game_id", "team", "player_name_norm"],
                    how="left",
                )
            else:
                player_stats["status_bucket"] = np.nan
                player_stats["practice_status"] = np.nan

            if not depth_charts.empty:
                depth_latest = depth_charts.copy()
                depth_latest["team"] = depth_latest["team"].apply(normalize_team_abbr)
                depth_latest["position"] = depth_latest["position"].astype(str).str.upper().str.strip()
                depth_latest["player_name_norm"] = depth_latest["player_name"].apply(
                    normalize_player_name
                )
                depth_latest = depth_latest[depth_latest["player_name_norm"] != ""]
                depth_latest["updated_at"] = pd.to_datetime(
                    depth_latest["updated_at"], errors="coerce"
                )
                depth_latest = depth_latest.sort_values("updated_at")
                depth_latest["rank"] = depth_latest["rank"].apply(parse_depth_rank)
                depth_latest = depth_latest.drop_duplicates(
                    subset=["team", "position", "player_name_norm"], keep="last"
                )
                player_stats = player_stats.merge(
                    depth_latest[["team", "position", "player_name_norm", "rank"]],
                    on=["team", "position", "player_name_norm"],
                    how="left",
                )
                player_stats = player_stats.rename(columns={"rank": "depth_rank"})
            else:
                player_stats["depth_rank"] = np.nan

            player_stats["status_bucket"] = (
                player_stats["status_bucket"].fillna("other").apply(normalize_injury_status)
            )
            player_stats["practice_status"] = (
                player_stats["practice_status"].fillna("available").apply(normalize_practice_status)
            )
            player_stats["injury_priority"] = player_stats["status_bucket"].map(
                INJURY_STATUS_PRIORITY
            ).fillna(INJURY_STATUS_PRIORITY.get("other", 1))
            player_stats["practice_priority"] = player_stats["practice_status"].map(
                PRACTICE_STATUS_PRIORITY
            ).fillna(1)
            player_stats["depth_rank"] = pd.to_numeric(
                player_stats["depth_rank"], errors="coerce"
            )
            player_stats["depth_rank"] = player_stats["depth_rank"].fillna(99.0)
            player_stats = player_stats.drop(columns=["player_name_norm"], errors="ignore")

            def add_dataset(target: str, positions: Iterable[str]) -> None:
                subset = player_stats[player_stats["position"].isin(list(positions))].copy()
                subset = subset[subset[target].notna()]
                if subset.empty:
                    logging.debug(
                        "Skipping %s dataset because no rows remained after filtering", target
                    )
                    return
                datasets[target] = subset

            ordered_targets = [
                "passing_yards",
                "passing_tds",
                "rushing_yards",
                "rushing_tds",
                "receiving_yards",
                "receptions",
                "receiving_tds",
            ]

            for target in ordered_targets:
                positions = TARGET_ALLOWED_POSITIONS.get(target)
                if not positions:
                    continue
                add_dataset(target, positions)

        if self.team_strength_frame is None:
            self.team_strength_frame = team_strength
        if self.team_strength_frame is not None and not self.team_strength_frame.empty:
            sorted_strength = self.team_strength_frame.sort_values(["team", "season", "week"])
            self.team_strength_latest_by_season = sorted_strength.drop_duplicates(
                subset=["team", "season"], keep="last"
            )
            self.team_strength_latest_overall = sorted_strength.drop_duplicates(
                subset=["team"], keep="last"
            )
        else:
            empty_strength = team_strength.iloc[0:0]
            self.team_strength_latest_by_season = empty_strength
            self.team_strength_latest_overall = empty_strength

        if self.context_feature_frame is None:
            self.context_feature_frame = pd.DataFrame(
                columns=[
                    "team",
                    "venue",
                    "day_of_week",
                    "referee",
                    "avg_rush_yards",
                    "avg_rec_yards",
                    "avg_receptions",
                    "avg_rush_tds",
                    "avg_rec_tds",
                ]
            )

        if player_stats.empty:
            self.player_feature_frame = pd.DataFrame(columns=player_stats.columns)
        else:
            self.player_feature_frame = player_stats

        home_strength = team_strength.rename(
            columns={
                "team": "home_team",
                "offense_pass_rating": "home_offense_pass_rating",
                "offense_rush_rating": "home_offense_rush_rating",
                "defense_pass_rating": "home_defense_pass_rating",
                "defense_rush_rating": "home_defense_rush_rating",
                "pace_seconds_per_play": "home_pace_seconds_per_play",
                "offense_epa": "home_offense_epa",
                "defense_epa": "home_defense_epa",
                "offense_success_rate": "home_offense_success_rate",
                "defense_success_rate": "home_defense_success_rate",
                "travel_penalty": "home_travel_penalty",
                "rest_penalty": "home_rest_penalty",
                "weather_adjustment": "home_weather_adjustment",
                "avg_timezone_diff_hours": "home_timezone_diff_hours",
            }
        )
        away_strength = team_strength.rename(
            columns={
                "team": "away_team",
                "offense_pass_rating": "away_offense_pass_rating",
                "offense_rush_rating": "away_offense_rush_rating",
                "defense_pass_rating": "away_defense_pass_rating",
                "defense_rush_rating": "away_defense_rush_rating",
                "pace_seconds_per_play": "away_pace_seconds_per_play",
                "offense_epa": "away_offense_epa",
                "defense_epa": "away_defense_epa",
                "offense_success_rate": "away_offense_success_rate",
                "defense_success_rate": "away_defense_success_rate",
                "travel_penalty": "away_travel_penalty",
                "rest_penalty": "away_rest_penalty",
                "weather_adjustment": "away_weather_adjustment",
                "avg_timezone_diff_hours": "away_timezone_diff_hours",
            }
        )

        home_history = team_game_history[team_game_history["is_home"]].drop(
            columns=["team", "is_home"]
        )
        home_history = home_history.rename(
            columns={
                "game_id": "game_id",
                "rolling_points_for": "home_points_for_avg",
                "rolling_points_against": "home_points_against_avg",
                "rolling_point_diff": "home_point_diff_avg",
                "rolling_win_pct": "home_win_pct_recent",
                "prev_points_for": "home_prev_points_for",
                "prev_points_against": "home_prev_points_against",
                "prev_point_diff": "home_prev_point_diff",
                "rest_days": "home_rest_days",
                "rest_penalty": "home_rest_penalty",
                "travel_penalty": "home_travel_penalty_hist",
                "timezone_diff_hours": "home_timezone_diff_hours",
            }
        )

        away_history = team_game_history[~team_game_history["is_home"]].drop(
            columns=["team", "is_home"]
        )
        away_history = away_history.rename(
            columns={
                "game_id": "game_id",
                "rolling_points_for": "away_points_for_avg",
                "rolling_points_against": "away_points_against_avg",
                "rolling_point_diff": "away_point_diff_avg",
                "rolling_win_pct": "away_win_pct_recent",
                "prev_points_for": "away_prev_points_for",
                "prev_points_against": "away_prev_points_against",
                "prev_point_diff": "away_prev_point_diff",
                "rest_days": "away_rest_days",
                "rest_penalty": "away_rest_penalty",
                "travel_penalty": "away_travel_penalty_hist",
                "timezone_diff_hours": "away_timezone_diff_hours",
            }
        )

        games_context = (
            games.merge(
                home_strength,
                on=["home_team", "season", "week"],
                how="left",
            )
            .merge(
                away_strength,
                on=["away_team", "season", "week"],
                how="left",
            )
            .merge(home_history, on="game_id", how="left")
            .merge(away_history, on="game_id", how="left")
        )

        if "home_travel_penalty_hist" in games_context.columns:
            games_context["home_travel_penalty"] = games_context["home_travel_penalty"].combine_first(
                games_context["home_travel_penalty_hist"]
            )
            games_context.drop(columns=["home_travel_penalty_hist"], inplace=True)
        if "away_travel_penalty_hist" in games_context.columns:
            games_context["away_travel_penalty"] = games_context["away_travel_penalty"].combine_first(
                games_context["away_travel_penalty_hist"]
            )
            games_context.drop(columns=["away_travel_penalty_hist"], inplace=True)

        games_context["moneyline_diff"] = games_context["home_moneyline"] - games_context["away_moneyline"]
        games_context["implied_prob_diff"] = (
            games_context["home_implied_prob"] - games_context["away_implied_prob"]
        )
        games_context["implied_prob_sum"] = (
            games_context["home_implied_prob"] + games_context["away_implied_prob"]
        )

        games_context["point_diff"] = games_context["home_score"] - games_context["away_score"]
        games_labeled = games_context.dropna(subset=["home_score", "away_score"])
        if games_labeled.empty:
            logging.warning(
                "No completed games with scores available. Game outcome model will be skipped."
            )
        else:
            datasets["game_outcome"] = games_labeled

        return datasets

    # ------------------------------------------------------------------
    # Upcoming feature preparation
    # ------------------------------------------------------------------

    def _get_latest_team_strength(self, team: str, season: Optional[str]) -> Optional[pd.Series]:
        if self.team_strength_latest_by_season is not None and not self.team_strength_latest_by_season.empty:
            if season:
                match = self.team_strength_latest_by_season[
                    (self.team_strength_latest_by_season["team"] == team)
                    & (self.team_strength_latest_by_season["season"] == season)
                ]
                if not match.empty:
                    return match.iloc[0]
        if self.team_strength_latest_overall is not None and not self.team_strength_latest_overall.empty:
            match = self.team_strength_latest_overall[
                self.team_strength_latest_overall["team"] == team
            ]
            if not match.empty:
                return match.iloc[0]
        return None

    def _get_latest_team_history(self, team: str, season: Optional[str]) -> Optional[pd.Series]:
        if self.team_history_latest_by_season is not None and not self.team_history_latest_by_season.empty:
            if season:
                match = self.team_history_latest_by_season[
                    (self.team_history_latest_by_season["team"] == team)
                    & (self.team_history_latest_by_season["season"] == season)
                ]
                if not match.empty:
                    return match.iloc[0]
        if self.team_history_latest_overall is not None and not self.team_history_latest_overall.empty:
            match = self.team_history_latest_overall[
                self.team_history_latest_overall["team"] == team
            ]
            if not match.empty:
                return match.iloc[0]
        return None

    def prepare_upcoming_game_features(self, upcoming_games: pd.DataFrame) -> pd.DataFrame:
        if upcoming_games.empty:
            return upcoming_games.copy()

        features = upcoming_games.copy()
        features["start_time"] = pd.to_datetime(features["start_time"])
        if "day_of_week" not in features.columns or features["day_of_week"].isna().any():
            features["day_of_week"] = features["start_time"].dt.day_name()

        numeric_placeholders = {
            "home_offense_pass_rating": np.nan,
            "home_offense_rush_rating": np.nan,
            "home_defense_pass_rating": np.nan,
            "home_defense_rush_rating": np.nan,
            "home_pace_seconds_per_play": np.nan,
            "home_offense_epa": np.nan,
            "home_defense_epa": np.nan,
            "home_offense_success_rate": np.nan,
            "home_defense_success_rate": np.nan,
            "home_travel_penalty": np.nan,
            "home_rest_penalty": np.nan,
            "home_weather_adjustment": np.nan,
            "home_timezone_diff_hours": np.nan,
            "away_offense_pass_rating": np.nan,
            "away_offense_rush_rating": np.nan,
            "away_defense_pass_rating": np.nan,
            "away_defense_rush_rating": np.nan,
            "away_pace_seconds_per_play": np.nan,
            "away_offense_epa": np.nan,
            "away_defense_epa": np.nan,
            "away_offense_success_rate": np.nan,
            "away_defense_success_rate": np.nan,
            "away_travel_penalty": np.nan,
            "away_rest_penalty": np.nan,
            "away_weather_adjustment": np.nan,
            "away_timezone_diff_hours": np.nan,
            "home_points_for_avg": np.nan,
            "home_points_against_avg": np.nan,
            "home_point_diff_avg": np.nan,
            "home_win_pct_recent": np.nan,
            "home_prev_points_for": np.nan,
            "home_prev_points_against": np.nan,
            "home_prev_point_diff": np.nan,
            "home_rest_days": np.nan,
            "home_injury_total": 0.0,
            "away_points_for_avg": np.nan,
            "away_points_against_avg": np.nan,
            "away_point_diff_avg": np.nan,
            "away_win_pct_recent": np.nan,
            "away_prev_points_for": np.nan,
            "away_prev_points_against": np.nan,
            "away_prev_point_diff": np.nan,
            "away_rest_days": np.nan,
            "away_injury_total": 0.0,
            "wind_mph": np.nan,
            "humidity": np.nan,
        }
        for col, default in numeric_placeholders.items():
            if col not in features.columns:
                features[col] = default

        for idx, row in features.iterrows():
            season = row.get("season")
            home_team = row.get("home_team")
            away_team = row.get("away_team")

            if home_team:
                strength = self._get_latest_team_strength(home_team, season)
                if strength is not None:
                    features.at[idx, "home_offense_pass_rating"] = strength.get("offense_pass_rating")
                    features.at[idx, "home_offense_rush_rating"] = strength.get("offense_rush_rating")
                    features.at[idx, "home_defense_pass_rating"] = strength.get("defense_pass_rating")
                    features.at[idx, "home_defense_rush_rating"] = strength.get("defense_rush_rating")
                    features.at[idx, "home_pace_seconds_per_play"] = strength.get("pace_seconds_per_play")
                    features.at[idx, "home_offense_epa"] = strength.get("offense_epa")
                    features.at[idx, "home_defense_epa"] = strength.get("defense_epa")
                    features.at[idx, "home_offense_success_rate"] = strength.get("offense_success_rate")
                    features.at[idx, "home_defense_success_rate"] = strength.get("defense_success_rate")
                    features.at[idx, "home_travel_penalty"] = strength.get("travel_penalty")
                    features.at[idx, "home_rest_penalty"] = strength.get("rest_penalty")
                    features.at[idx, "home_weather_adjustment"] = strength.get("weather_adjustment")
                    features.at[idx, "home_timezone_diff_hours"] = strength.get("avg_timezone_diff_hours")
                history = self._get_latest_team_history(home_team, season)
                if history is not None:
                    features.at[idx, "home_points_for_avg"] = history.get("rolling_points_for")
                    features.at[idx, "home_points_against_avg"] = history.get("rolling_points_against")
                    features.at[idx, "home_point_diff_avg"] = history.get("rolling_point_diff")
                    features.at[idx, "home_win_pct_recent"] = history.get("rolling_win_pct")
                    features.at[idx, "home_prev_points_for"] = history.get("prev_points_for")
                    features.at[idx, "home_prev_points_against"] = history.get("prev_points_against")
                    features.at[idx, "home_prev_point_diff"] = history.get("prev_point_diff")
                    features.at[idx, "home_rest_days"] = history.get("rest_days")
                    if pd.isna(features.at[idx, "home_rest_penalty"]):
                        features.at[idx, "home_rest_penalty"] = history.get("rest_penalty")
                    if pd.isna(features.at[idx, "home_travel_penalty"]):
                        features.at[idx, "home_travel_penalty"] = history.get("travel_penalty")
                    if pd.isna(features.at[idx, "home_timezone_diff_hours"]):
                        features.at[idx, "home_timezone_diff_hours"] = history.get("timezone_diff_hours")

            if away_team:
                strength = self._get_latest_team_strength(away_team, season)
                if strength is not None:
                    features.at[idx, "away_offense_pass_rating"] = strength.get("offense_pass_rating")
                    features.at[idx, "away_offense_rush_rating"] = strength.get("offense_rush_rating")
                    features.at[idx, "away_defense_pass_rating"] = strength.get("defense_pass_rating")
                    features.at[idx, "away_defense_rush_rating"] = strength.get("defense_rush_rating")
                    features.at[idx, "away_pace_seconds_per_play"] = strength.get("pace_seconds_per_play")
                    features.at[idx, "away_offense_epa"] = strength.get("offense_epa")
                    features.at[idx, "away_defense_epa"] = strength.get("defense_epa")
                    features.at[idx, "away_offense_success_rate"] = strength.get("offense_success_rate")
                    features.at[idx, "away_defense_success_rate"] = strength.get("defense_success_rate")
                    features.at[idx, "away_travel_penalty"] = strength.get("travel_penalty")
                    features.at[idx, "away_rest_penalty"] = strength.get("rest_penalty")
                    features.at[idx, "away_weather_adjustment"] = strength.get("weather_adjustment")
                    features.at[idx, "away_timezone_diff_hours"] = strength.get("avg_timezone_diff_hours")
                history = self._get_latest_team_history(away_team, season)
                if history is not None:
                    features.at[idx, "away_points_for_avg"] = history.get("rolling_points_for")
                    features.at[idx, "away_points_against_avg"] = history.get("rolling_points_against")
                    features.at[idx, "away_point_diff_avg"] = history.get("rolling_point_diff")
                    features.at[idx, "away_win_pct_recent"] = history.get("rolling_win_pct")
                    features.at[idx, "away_prev_points_for"] = history.get("prev_points_for")
                    features.at[idx, "away_prev_points_against"] = history.get("prev_points_against")
                    features.at[idx, "away_prev_point_diff"] = history.get("prev_point_diff")
                    features.at[idx, "away_rest_days"] = history.get("rest_days")
                    if pd.isna(features.at[idx, "away_rest_penalty"]):
                        features.at[idx, "away_rest_penalty"] = history.get("rest_penalty")
                    if pd.isna(features.at[idx, "away_travel_penalty"]):
                        features.at[idx, "away_travel_penalty"] = history.get("travel_penalty")
                    if pd.isna(features.at[idx, "away_timezone_diff_hours"]):
                        features.at[idx, "away_timezone_diff_hours"] = history.get("timezone_diff_hours")

        fill_defaults = {col: 0.0 for col in numeric_placeholders.keys()}
        features[list(fill_defaults.keys())] = features[list(fill_defaults.keys())].fillna(fill_defaults)

        features["moneyline_diff"] = features["home_moneyline"] - features["away_moneyline"]
        features["implied_prob_diff"] = features["home_implied_prob"] - features["away_implied_prob"]
        features["implied_prob_sum"] = features["home_implied_prob"] + features["away_implied_prob"]

        return features

    def prepare_upcoming_player_features(
        self,
        upcoming_games: pd.DataFrame,
        starters_per_position: Optional[Dict[str, int]] = None,
        lineup_rows: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        base_players = self.player_feature_frame

        if base_players is None:
            return pd.DataFrame()

        if (
            base_players.empty
            or upcoming_games.empty
        ):
            return pd.DataFrame()

        if starters_per_position is None:
            starters_per_position = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}

        position_groups = {
            "QB": {"QB"},
            "RB": {"RB", "HB", "FB"},
            "WR": {"WR"},
            "TE": {"TE"},
        }

        def _is_stale_lineup_entry(row: Any, now: Optional[Union[str, dt.datetime]] = None) -> bool:
            if isinstance(row, pd.Series):
                series = row
            elif isinstance(row, dict):
                series = pd.Series(row)
            elif hasattr(row, "_asdict"):
                series = pd.Series(row._asdict())
            else:
                series = pd.Series({"updated_at": row})

            now_ts = (
                pd.Timestamp.now(tz="UTC")
                if now is None
                else pd.to_datetime(now, utc=True, errors="coerce")
            )
            if pd.isna(now_ts):
                now_ts = pd.Timestamp.now(tz="UTC")

            updated_at = pd.to_datetime(series.get("updated_at"), utc=True, errors="coerce")
            if pd.isna(updated_at):
                return False

            game_start = pd.to_datetime(series.get("game_start"), utc=True, errors="coerce")
            if not pd.isna(game_start):
                return updated_at < (
                    game_start - pd.Timedelta(days=LINEUP_MAX_AGE_BEFORE_GAME_DAYS)
                )

            return (now_ts - updated_at) > pd.Timedelta(days=LINEUP_STALENESS_DAYS)

        base_players = base_players.copy()

        latest_players = (
            base_players.sort_values("start_time")
            .groupby("player_id", as_index=False)
            .tail(1)
        )

        latest_players = latest_players.drop(columns=["player_key"], errors="ignore")

        if "position" in latest_players.columns:
            latest_players["position"] = latest_players["position"].apply(
                normalize_position
            )

        season_source = base_players.copy()
        season_source["team"] = season_source["team"].apply(normalize_team_abbr)
        season_source = season_source[season_source["team"].isin(TEAM_ABBR_CANONICAL.keys())]

        if not season_source.empty:
            aggregate_candidates = [
                "passing_attempts",
                "passing_yards",
                "passing_tds",
                "rushing_attempts",
                "rushing_yards",
                "rushing_tds",
                "receiving_targets",
                "receiving_yards",
                "receptions",
                "receiving_tds",
                "fantasy_points",
                "snap_count",
            ]
            present_metrics = [
                col for col in aggregate_candidates if col in season_source.columns
            ]
            if present_metrics:
                season_totals = (
                    season_source.groupby(["player_id", "team"], as_index=False)[
                        present_metrics
                    ].sum(min_count=1)
                )
                rename_map = {col: f"season_{col}" for col in present_metrics}
                season_totals = season_totals.rename(columns=rename_map)
                latest_players = latest_players.merge(
                    season_totals,
                    on=["player_id", "team"],
                    how="left",
                )

        injuries_latest = pd.DataFrame()
        if self.injury_frame is not None and not self.injury_frame.empty:
            injuries_latest = self.injury_frame.copy()
            injuries_latest["team"] = injuries_latest["team"].apply(normalize_team_abbr)
            injuries_latest["player_name_norm"] = injuries_latest["player_name"].apply(
                normalize_player_name
            )
            injuries_latest = injuries_latest[injuries_latest["player_name_norm"] != ""]

            if "status_original" not in injuries_latest.columns:
                injuries_latest["status_original"] = injuries_latest.get("status")
            injuries_latest["status_bucket"] = injuries_latest.apply(
                lambda row: compute_injury_bucket(
                    row.get("status_original") or row.get("status"),
                    row.get("description"),
                ),
                axis=1,
            )

            practice_source = injuries_latest.get("practice_status_raw")
            if practice_source is None:
                practice_source = injuries_latest.get("practice_status")
            if practice_source is None:
                practice_source = pd.Series("", index=injuries_latest.index)
            injuries_latest["_has_practice_report"] = (
                practice_source.astype(str).str.strip() != ""
            )
            injuries_latest["practice_status"] = practice_source.apply(
                normalize_practice_status
            )

            injuries_latest["report_time"] = pd.to_datetime(
                injuries_latest["report_time"], errors="coerce"
            )
            injuries_latest = injuries_latest.sort_values("report_time")
            injuries_latest = injuries_latest.drop_duplicates(
                subset=["team", "player_name_norm"], keep="last"
            )

            active_mask = (
                injuries_latest["_has_practice_report"]
                & injuries_latest["practice_status"].isin({"full", "available", "rest"})
            )
            injuries_latest.loc[
                active_mask & ~injuries_latest["status_bucket"].isin({"suspended"}),
                "status_bucket",
            ] = "other"
            injuries_latest = injuries_latest.drop(columns=["_has_practice_report"], errors="ignore")

        depth_latest = pd.DataFrame()
        if self.depth_chart_frame is not None and not self.depth_chart_frame.empty:
            depth_latest = self.depth_chart_frame.copy()
            depth_latest["team"] = depth_latest["team"].apply(normalize_team_abbr)
            depth_latest["position"] = depth_latest["position"].apply(normalize_position)
            depth_latest["player_name_norm"] = depth_latest["player_name"].apply(
                normalize_player_name
            )
            depth_latest = depth_latest[
                (depth_latest["player_name_norm"] != "")
                & (depth_latest["position"] != "")
            ]
            depth_latest["updated_at"] = pd.to_datetime(
                depth_latest["updated_at"], errors="coerce"
            )
            depth_latest["rank"] = depth_latest["rank"].apply(parse_depth_rank)
            if "depth_id" in depth_latest.columns:
                depth_latest["_lineup_entry"] = depth_latest["depth_id"].astype(str).str.startswith(
                    "msf-lineup:"
                )
            else:
                depth_latest["_lineup_entry"] = False

        if lineup_rows is not None and not lineup_rows.empty:
            lineup_latest = lineup_rows.copy()
            lineup_latest["team"] = lineup_latest["team"].apply(normalize_team_abbr)
            lineup_latest["position"] = lineup_latest["position"].apply(normalize_position)
            lineup_latest["player_name_norm"] = lineup_latest["player_name"].apply(
                normalize_player_name
            )
            lineup_latest = lineup_latest[
                (lineup_latest["player_name_norm"] != "")
                & (lineup_latest["position"] != "")
            ]
            lineup_latest["updated_at"] = pd.to_datetime(
                lineup_latest["updated_at"], errors="coerce"
            )
            if "game_start" in lineup_latest.columns:
                lineup_latest["game_start"] = pd.to_datetime(
                    lineup_latest["game_start"], errors="coerce", utc=True
                )
            lineup_latest["rank"] = lineup_latest["rank"].apply(parse_depth_rank)
            if "depth_id" in lineup_latest.columns:
                lineup_latest["_lineup_entry"] = True
            else:
                lineup_latest["depth_id"] = lineup_latest.apply(
                    lambda row: f"msf-lineup:{row['team']}:{row['position']}:{row['player_name_norm']}",
                    axis=1,
                )
                lineup_latest["_lineup_entry"] = True

            if depth_latest.empty:
                depth_latest = lineup_latest
            else:
                frames = [
                    frame
                    for frame in (depth_latest, lineup_latest)
                    if frame is not None and not frame.empty
                ]
                if frames:
                    depth_latest = safe_concat(frames, ignore_index=True, sort=False)

        if not depth_latest.empty:
            lineup_entry_mask = depth_latest.get("_lineup_entry")
            if lineup_entry_mask is not None:
                lineup_entry_mask = lineup_entry_mask.fillna(False)
                lineup_entry_mask = lineup_entry_mask.infer_objects(copy=False)
                lineup_entry_mask = lineup_entry_mask.astype(bool)
                if lineup_entry_mask.any():
                    stale_entries = depth_latest.apply(_is_stale_lineup_entry, axis=1)
                    stale_mask = lineup_entry_mask & stale_entries
                    if stale_mask.any():
                        logging.debug(
                            "Discarding %d stale lineup entries based on age relative to kickoff",
                            int(stale_mask.sum()),
                        )
                        depth_latest = depth_latest.loc[~stale_mask].copy()

            depth_latest = depth_latest.sort_values(
                ["_lineup_entry", "updated_at"], ascending=[True, True]
            )
            depth_latest = depth_latest.drop_duplicates(
                subset=["team", "position", "player_name_norm"], keep="last"
            )
            if "depth_id" in depth_latest.columns:
                depth_latest["_lineup_entry"] = depth_latest["depth_id"].astype(str).str.startswith(
                    "msf-lineup:"
                )
            else:
                depth_latest["_lineup_entry"] = False

        latest_players["player_name_norm"] = latest_players["player_name"].apply(
            normalize_player_name
        )
        latest_players["__pname_key"] = latest_players["player_name"].map(
            robust_player_name_key
        )

        if "depth_rank" not in latest_players.columns:
            latest_players["depth_rank"] = np.nan

        if "status_bucket" not in latest_players.columns:
            latest_players["status_bucket"] = np.nan
        if "practice_status" not in latest_players.columns:
            latest_players["practice_status"] = np.nan

        template_columns = latest_players.columns.tolist()

        existing_keys: set[Tuple[str, str]] = set(
            zip(latest_players["team"], latest_players["player_name_norm"])
        )

        additional_rows: List[Dict[str, Any]] = []

        if not depth_latest.empty:
            for depth_row in depth_latest.itertuples():
                key = (depth_row.team, depth_row.player_name_norm)
                if key in existing_keys:
                    continue

                row_series = pd.Series(depth_row._asdict())
                lineup_flag = row_series.get("_lineup_entry")
                if (
                    pd.notna(lineup_flag)
                    and bool(lineup_flag)
                    and _is_stale_lineup_entry(row_series)
                ):
                    continue

                placeholder: Dict[str, Any] = {
                    col: np.nan for col in template_columns if col not in {"player_id", "team", "position"}
                }
                placeholder.setdefault("status_bucket", "other")
                placeholder.setdefault("practice_status", "available")
                placeholder.setdefault(
                    "injury_priority", INJURY_STATUS_PRIORITY.get("other", 1)
                )

                # Ensure base identifiers exist even when the player has not logged stats yet.
                placeholder.update(
                    {
                        "player_id": f"depth_{depth_row.team}_{depth_row.player_name_norm}",
                        "player_name": getattr(depth_row, "player_name", ""),
                        "team": depth_row.team,
                        "position": depth_row.position,
                        "player_name_norm": depth_row.player_name_norm,
                        "depth_rank": getattr(depth_row, "rank", np.nan),
                    }
                )

                season_metric_defaults = {
                    col: 0.0
                    for col in template_columns
                    if col.startswith("season_")
                }
                placeholder.update(season_metric_defaults)

                additional_rows.append(placeholder)

        if additional_rows:
            additional_df = pd.DataFrame(additional_rows)
            missing_cols = set(template_columns) - set(additional_df.columns)
            for col in missing_cols:
                additional_df[col] = np.nan
            frames = [
                frame
                for frame in (
                    latest_players,
                    additional_df[template_columns],
                )
                if frame is not None and not frame.empty
            ]
            if frames:
                latest_players = safe_concat(
                    frames,
                    ignore_index=True,
                    sort=False,
                )
            existing_keys = set(
                zip(latest_players["team"], latest_players["player_name_norm"])
            )

        if not injuries_latest.empty:
            injury_placeholders: List[Dict[str, Any]] = []
            for injury_row in injuries_latest.itertuples():
                key = (injury_row.team, injury_row.player_name_norm)
                if key in existing_keys:
                    continue
                status_bucket = getattr(injury_row, "status_bucket", "other")
                if status_bucket in INACTIVE_INJURY_BUCKETS:
                    continue
                position = normalize_position(getattr(injury_row, "position", ""))
                if not position:
                    continue
                placeholder: Dict[str, Any] = {
                    col: np.nan
                    for col in template_columns
                    if col not in {"player_id", "team", "position"}
                }
                placeholder.update(
                    {
                        "player_id": f"injury_{injury_row.team}_{injury_row.player_name_norm}",
                        "player_name": getattr(injury_row, "player_name", ""),
                        "team": injury_row.team,
                        "position": position,
                        "player_name_norm": injury_row.player_name_norm,
                        "status_bucket": status_bucket,
                        "practice_status": getattr(
                            injury_row, "practice_status", "available"
                        ),
                    }
                )
                season_metric_defaults = {
                    col: 0.0
                    for col in template_columns
                    if col.startswith("season_")
                }
                placeholder.update(season_metric_defaults)
                placeholder.setdefault(
                    "injury_priority",
                    INJURY_STATUS_PRIORITY.get(
                        status_bucket, INJURY_STATUS_PRIORITY.get("other", 1)
                    ),
                )
                injury_placeholders.append(placeholder)

            if injury_placeholders:
                injury_df = pd.DataFrame(injury_placeholders)
                missing_cols = set(template_columns) - set(injury_df.columns)
                for col in missing_cols:
                    injury_df[col] = np.nan
                latest_players = safe_concat(
                    [latest_players, injury_df[template_columns]],
                    ignore_index=True,
                    sort=False,
                )
                existing_keys = set(
                    zip(latest_players["team"], latest_players["player_name_norm"])
                )

        if not injuries_latest.empty:
            latest_players = latest_players.merge(
                injuries_latest[
                    ["team", "player_name_norm", "status_bucket", "status", "practice_status"]
                ],
                on=["team", "player_name_norm"],
                how="left",
                suffixes=("", "_inj"),
            )

            if "status_bucket_inj" in latest_players.columns:
                latest_players["status_bucket"] = latest_players[
                    "status_bucket_inj"
                ].combine_first(latest_players.get("status_bucket"))
                latest_players = latest_players.drop(
                    columns=["status_bucket_inj"], errors="ignore"
                )

            if "practice_status_inj" in latest_players.columns:
                latest_players["practice_status"] = latest_players[
                    "practice_status_inj"
                ].combine_first(latest_players.get("practice_status"))
                latest_players = latest_players.drop(
                    columns=["practice_status_inj"], errors="ignore"
                )

            if "status_inj" in latest_players.columns and "status" not in latest_players:
                latest_players = latest_players.rename(
                    columns={"status_inj": "status"}
                )
        else:
            latest_players["status_bucket"] = latest_players.get("status_bucket", np.nan)
            latest_players["practice_status"] = latest_players.get(
                "practice_status", np.nan
            )

        if not depth_latest.empty:
            merge_columns = ["team", "position", "player_name_norm", "rank"]
            for optional in ("_lineup_entry", "depth_id", "updated_at"):
                if optional in depth_latest.columns:
                    merge_columns.append(optional)

            latest_players = latest_players.merge(
                depth_latest[merge_columns],
                on=["team", "position", "player_name_norm"],
                how="left",
                suffixes=("", "_depth"),
            )

            depth_rank_sources = [
                col
                for col in ("depth_rank", "rank_depth", "rank")
                if col in latest_players.columns
            ]

            if depth_rank_sources:
                latest_players["depth_rank"] = (
                    latest_players[depth_rank_sources]
                    .bfill(axis=1)
                    .iloc[:, 0]
                )
            else:
                latest_players["depth_rank"] = np.nan

            columns_to_drop = [
                col
                for col in ("rank_depth", "rank", "depth_id", "updated_at")
                if col in latest_players.columns and col != "depth_rank"
            ]
            if columns_to_drop:
                latest_players = latest_players.drop(
                    columns=columns_to_drop, errors="ignore"
                )
        else:
            latest_players["depth_rank"] = latest_players.get("depth_rank", np.nan)

        latest_players = ensure_lineup_players_in_latest(latest_players, lineup_rows)

        latest_players["status_bucket"] = (
            latest_players["status_bucket"].fillna("other").apply(normalize_injury_status)
        )
        latest_players["practice_status"] = (
            latest_players["practice_status"].fillna("available").apply(normalize_practice_status)
        )
        latest_players["injury_priority"] = latest_players["status_bucket"].map(
            INJURY_STATUS_PRIORITY
        ).fillna(INJURY_STATUS_PRIORITY.get("other", 1))
        latest_players["practice_priority"] = latest_players["practice_status"].map(
            PRACTICE_STATUS_PRIORITY
        ).fillna(1)
        latest_players["depth_rank"] = pd.to_numeric(
            latest_players["depth_rank"], errors="coerce"
        )
        if "_lineup_entry" in latest_players.columns:
            lineup_mask = latest_players["_lineup_entry"].infer_objects(copy=False)
            lineup_mask = coerce_boolean_mask(lineup_mask)

            if lineup_mask.dtype != bool:
                lineup_mask = lineup_mask.astype(bool)
            starter_allowance = (
                latest_players["position"].fillna("").map(starters_per_position).fillna(1)
            )
            starter_mask = lineup_mask & (
                latest_players["depth_rank"].isna()
                | (latest_players["depth_rank"] <= starter_allowance)
            )
            latest_players["is_projected_starter"] = starter_mask
            latest_players = latest_players.drop(
                columns=["_lineup_entry"], errors="ignore"
            )
        else:
            latest_players["is_projected_starter"] = False
        latest_players["depth_rank"] = latest_players["depth_rank"].fillna(99.0)

        games = upcoming_games.copy()
        games["start_time"] = pd.to_datetime(games["start_time"], utc=True, errors="coerce")
        games = games[games["start_time"].notna()]
        eastern = ZoneInfo("America/New_York")
        if "local_start_time" in games.columns:
            games["local_start_time"] = pd.to_datetime(
                games["local_start_time"], utc=True, errors="coerce"
            ).dt.tz_convert(eastern)
        else:
            games["local_start_time"] = games["start_time"].dt.tz_convert(eastern)

        if "day_of_week" in games.columns:
            games["day_of_week"] = games["day_of_week"].where(
                games["day_of_week"].notna(), games["local_start_time"].dt.day_name()
            )
        else:
            games["day_of_week"] = games["local_start_time"].dt.day_name()
        games["home_team"] = games["home_team"].apply(normalize_team_abbr)
        games["away_team"] = games["away_team"].apply(normalize_team_abbr)

        selected_rows: List[pd.Series] = []

        for game in games.itertuples():
            game_id = getattr(game, "game_id")
            season = getattr(game, "season", None)
            week = getattr(game, "week", None)
            start_time = getattr(game, "start_time")
            venue = getattr(game, "venue", None)
            city = getattr(game, "city", None)
            state = getattr(game, "state", None)
            day_of_week = getattr(game, "day_of_week", None)
            referee = getattr(game, "referee", None)
            weather = getattr(game, "weather_conditions", None)
            temperature = getattr(game, "temperature_f", None)
            home_team = getattr(game, "home_team", None)
            away_team = getattr(game, "away_team", None)

            for team, opponent in ((away_team, home_team), (home_team, away_team)):
                if not team or not opponent:
                    continue

                candidates = latest_players[latest_players["team"] == team]
                if candidates.empty:
                    continue

                chosen_players: List[pd.Series] = []
                used_player_ids: set[str] = set()

                for pos_key, count in starters_per_position.items():
                    allowed_positions = position_groups.get(pos_key, {pos_key})
                    position_candidates = candidates[
                        candidates["position"].isin(allowed_positions)
                    ]
                    if position_candidates.empty:
                        continue
                    position_candidates = position_candidates.copy()

                    if "status_bucket" in position_candidates.columns:
                        filtered_candidates = position_candidates[
                            ~position_candidates["status_bucket"].isin(INACTIVE_INJURY_BUCKETS)
                        ]
                        if not filtered_candidates.empty:
                            position_candidates = filtered_candidates

                    if "injury_priority" not in position_candidates.columns:
                        position_candidates["injury_priority"] = INJURY_STATUS_PRIORITY.get(
                            "other", 1
                        )
                    else:
                        position_candidates["injury_priority"] = position_candidates[
                            "injury_priority"
                        ].fillna(INJURY_STATUS_PRIORITY.get("other", 1))

                    if "is_projected_starter" not in position_candidates.columns:
                        position_candidates["is_projected_starter"] = False
                    else:
                        position_candidates["is_projected_starter"] = (
                            position_candidates["is_projected_starter"].fillna(False).astype(bool)
                        )

                    position_candidates["practice_status"] = position_candidates[
                        "practice_status"
                    ].apply(normalize_practice_status)
                    practice_priority = position_candidates["practice_status"].map(
                        PRACTICE_STATUS_PRIORITY
                    ).fillna(1)
                    position_candidates["practice_priority"] = practice_priority

                    if pos_key == "QB":
                        sort_cols = [
                            "season_passing_attempts",
                            "season_passing_yards",
                            "season_fantasy_points",
                            "season_snap_count",
                            "passing_attempts",
                            "passing_yards",
                            "fantasy_points",
                            "snap_count",
                        ]
                    elif pos_key == "RB":
                        sort_cols = [
                            "season_rushing_attempts",
                            "season_rushing_yards",
                            "season_receiving_targets",
                            "season_snap_count",
                            "season_fantasy_points",
                            "rushing_attempts",
                            "rushing_yards",
                            "receiving_targets",
                            "snap_count",
                            "fantasy_points",
                        ]
                    elif pos_key == "WR":
                        sort_cols = [
                            "season_receiving_targets",
                            "season_receiving_yards",
                            "season_receptions",
                            "season_fantasy_points",
                            "receiving_targets",
                            "receiving_yards",
                            "receptions",
                            "fantasy_points",
                        ]
                    elif pos_key == "TE":
                        sort_cols = [
                            "season_receiving_targets",
                            "season_receptions",
                            "season_receiving_yards",
                            "season_fantasy_points",
                            "receiving_targets",
                            "receptions",
                            "receiving_yards",
                            "fantasy_points",
                        ]
                    else:
                        sort_cols = [
                            "season_snap_count",
                            "season_fantasy_points",
                            "snap_count",
                            "fantasy_points",
                        ]

                    sort_columns: List[str] = []
                    ascending_flags: List[bool] = []

                    if "is_projected_starter" in position_candidates.columns:
                        position_candidates["is_projected_starter"] = (
                            position_candidates["is_projected_starter"].fillna(False).astype(bool)
                        )
                        sort_columns.append("is_projected_starter")
                        ascending_flags.append(False)

                    if "depth_rank" in position_candidates.columns:
                        sort_columns.append("depth_rank")
                        ascending_flags.append(True)

                    sort_columns.append("injury_priority")
                    ascending_flags.append(False)
                    sort_columns.append("practice_priority")
                    ascending_flags.append(False)

                    for col in sort_cols:
                        if col not in position_candidates.columns:
                            position_candidates[col] = 0.0
                        else:
                            position_candidates[col] = position_candidates[col].fillna(0.0)
                        sort_columns.append(col)
                        ascending_flags.append(False)

                    position_candidates = position_candidates.sort_values(
                        sort_columns, ascending=ascending_flags
                    )

                    starters_first = position_candidates[
                        position_candidates["is_projected_starter"]
                    ]
                    backups_after = position_candidates[
                        ~position_candidates["is_projected_starter"]
                    ]
                    pos_selected = 0
                    for pool in (starters_first, backups_after):
                        if pos_selected >= count:
                            break
                        for _, player_row in pool.iterrows():
                            player_id = player_row.get("player_id")
                            if player_id in used_player_ids:
                                continue
                            chosen_players.append(player_row)
                            used_player_ids.add(player_id)
                            pos_selected += 1
                            if pos_selected >= count:
                                break

                if not chosen_players:
                    continue

                for player_row in chosen_players:
                    row_copy = player_row.copy()
                    row_copy["game_id"] = game_id
                    row_copy["season"] = season
                    row_copy["week"] = week
                    row_copy["start_time"] = start_time
                    row_copy["local_start_time"] = getattr(game, "local_start_time", pd.NaT)
                    row_copy["venue"] = venue
                    row_copy["city"] = city
                    row_copy["state"] = state
                    row_copy["day_of_week"] = day_of_week
                    row_copy["referee"] = referee
                    row_copy["weather_conditions"] = weather
                    row_copy["temperature_f"] = temperature
                    row_copy["home_team"] = home_team
                    row_copy["away_team"] = away_team
                    row_copy["opponent"] = opponent

                    strength = self._get_latest_team_strength(team, season)
                    if strength is not None:
                        row_copy["offense_pass_rating"] = strength.get("offense_pass_rating")
                        row_copy["offense_rush_rating"] = strength.get("offense_rush_rating")
                        row_copy["defense_pass_rating"] = strength.get("defense_pass_rating")
                        row_copy["defense_rush_rating"] = strength.get("defense_rush_rating")
                        row_copy["pace_seconds_per_play"] = strength.get("pace_seconds_per_play")
                        row_copy["offense_epa"] = strength.get("offense_epa")
                        row_copy["defense_epa"] = strength.get("defense_epa")
                        row_copy["offense_success_rate"] = strength.get("offense_success_rate")
                        row_copy["defense_success_rate"] = strength.get("defense_success_rate")
                        row_copy["travel_penalty"] = strength.get("travel_penalty")
                        row_copy["rest_penalty"] = strength.get("rest_penalty")
                        row_copy["weather_adjustment"] = strength.get("weather_adjustment")
                        row_copy["avg_timezone_diff_hours"] = strength.get("avg_timezone_diff_hours")

                    opp_strength = self._get_latest_team_strength(opponent, season)
                    if opp_strength is not None:
                        row_copy["opp_offense_pass_rating"] = opp_strength.get("offense_pass_rating")
                        row_copy["opp_offense_rush_rating"] = opp_strength.get("offense_rush_rating")
                        row_copy["opp_defense_pass_rating"] = opp_strength.get("defense_pass_rating")
                        row_copy["opp_defense_rush_rating"] = opp_strength.get("defense_rush_rating")
                        row_copy["opp_pace_seconds_per_play"] = opp_strength.get("pace_seconds_per_play")
                        row_copy["opp_offense_epa"] = opp_strength.get("offense_epa")
                        row_copy["opp_defense_epa"] = opp_strength.get("defense_epa")
                        row_copy["opp_offense_success_rate"] = opp_strength.get("offense_success_rate")
                        row_copy["opp_defense_success_rate"] = opp_strength.get("defense_success_rate")
                        row_copy["opp_travel_penalty"] = opp_strength.get("travel_penalty")
                        row_copy["opp_rest_penalty"] = opp_strength.get("rest_penalty")
                        row_copy["opp_weather_adjustment"] = opp_strength.get("weather_adjustment")
                        row_copy["opp_timezone_diff_hours"] = opp_strength.get("avg_timezone_diff_hours")

                    selected_rows.append(row_copy)

        if not selected_rows:
            return pd.DataFrame()

        player_features = pd.DataFrame(selected_rows)

        # Merge contextual averages for updated venue/day/ref when available.
        if self.context_feature_frame is not None and not self.context_feature_frame.empty:
            contextual = self.context_feature_frame.rename(
                columns={
                    "avg_rush_yards": "avg_rush_yards_ctx",
                    "avg_rec_yards": "avg_rec_yards_ctx",
                    "avg_receptions": "avg_receptions_ctx",
                    "avg_rush_tds": "avg_rush_tds_ctx",
                    "avg_rec_tds": "avg_rec_tds_ctx",
                }
            )
            player_features = player_features.merge(
                contextual,
                on=["team", "venue", "day_of_week", "referee"],
                how="left",
            )
            for src, dest in [
                ("avg_rush_yards_ctx", "avg_rush_yards"),
                ("avg_rec_yards_ctx", "avg_rec_yards"),
                ("avg_receptions_ctx", "avg_receptions"),
                ("avg_rush_tds_ctx", "avg_rush_tds"),
                ("avg_rec_tds_ctx", "avg_rec_tds"),
            ]:
                if src in player_features.columns:
                    player_features[dest] = player_features[dest].where(
                        player_features[dest].notna(), player_features[src]
                    )
                    player_features = player_features.drop(columns=[src])

        player_features = player_features.drop(columns=["player_name_norm"], errors="ignore")

        return player_features

    def _compute_team_unit_strength(
        self, player_stats: pd.DataFrame, advanced_metrics: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        base_columns = [
            "season",
            "week",
            "team",
            "offense_pass_rating",
            "offense_rush_rating",
            "defense_pass_rating",
            "defense_rush_rating",
            "pace_seconds_per_play",
            "offense_epa",
            "defense_epa",
            "offense_success_rate",
            "defense_success_rate",
            "travel_penalty",
            "rest_penalty",
            "weather_adjustment",
            "avg_timezone_diff_hours",
        ]

        if player_stats.empty and (advanced_metrics is None or advanced_metrics.empty):
            return pd.DataFrame(columns=base_columns)

        stats = player_stats.copy()
        numeric_cols = [
            "rushing_yards",
            "rushing_attempts",
            "receiving_yards",
            "receiving_targets",
            "passing_yards",
            "passing_attempts",
            "rushing_tds",
            "passing_tds",
        ]
        for col in numeric_cols:
            if col not in stats.columns:
                stats[col] = 0.0
            stats[col] = stats[col].fillna(0.0)

        if "home_team" in stats.columns and "away_team" in stats.columns:
            stats = stats.copy()
            stats["opponent"] = np.where(
                stats["team"] == stats["home_team"], stats["away_team"], stats["home_team"]
            )
        else:
            stats["opponent"] = np.nan

        offense = (
            stats.dropna(subset=["team", "season", "week"])
            .groupby(["season", "week", "team"], as_index=False)
            .agg(
                rushing_yards=pd.NamedAgg(column="rushing_yards", aggfunc="sum"),
                rushing_attempts=pd.NamedAgg(column="rushing_attempts", aggfunc="sum"),
                receiving_yards=pd.NamedAgg(column="receiving_yards", aggfunc="sum"),
                receiving_targets=pd.NamedAgg(column="receiving_targets", aggfunc="sum"),
                passing_yards=pd.NamedAgg(column="passing_yards", aggfunc="sum"),
                passing_attempts=pd.NamedAgg(column="passing_attempts", aggfunc="sum"),
                rushing_tds=pd.NamedAgg(column="rushing_tds", aggfunc="sum"),
                passing_tds=pd.NamedAgg(column="passing_tds", aggfunc="sum"),
            )
        )

        defense = (
            stats.dropna(subset=["opponent", "season", "week"])
            .groupby(["season", "week", "opponent"], as_index=False)
            .agg(
                opp_rushing_yards=pd.NamedAgg(column="rushing_yards", aggfunc="sum"),
                opp_rushing_attempts=pd.NamedAgg(column="rushing_attempts", aggfunc="sum"),
                opp_receiving_yards=pd.NamedAgg(column="receiving_yards", aggfunc="sum"),
                opp_receiving_targets=pd.NamedAgg(column="receiving_targets", aggfunc="sum"),
                opp_passing_yards=pd.NamedAgg(column="passing_yards", aggfunc="sum"),
                opp_passing_attempts=pd.NamedAgg(column="passing_attempts", aggfunc="sum"),
            )
            .rename(columns={"opponent": "team"})
        )

        merged = offense.merge(defense, on=["season", "week", "team"], how="left")
        merged["plays"] = merged["rushing_attempts"] + merged["passing_attempts"]
        merged["yards_per_play"] = np.where(
            merged["plays"] > 0,
            (merged["rushing_yards"] + merged["passing_yards"]) / merged["plays"],
            np.nan,
        )
        merged["rush_per_attempt"] = np.where(
            merged["rushing_attempts"] > 0,
            merged["rushing_yards"] / merged["rushing_attempts"],
            np.nan,
        )
        merged["pass_per_attempt"] = np.where(
            merged["passing_attempts"] > 0,
            merged["passing_yards"] / merged["passing_attempts"],
            np.nan,
        )
        merged["pace_seconds_per_play"] = np.where(
            merged["plays"] > 0,
            3600.0 / merged["plays"],
            np.nan,
        )

        merged["offense_success_rate"] = np.clip(
            np.where(
                merged["plays"] > 0,
                (merged["rushing_yards"] + merged["passing_yards"]) / (merged["plays"] * 4.0),
                np.nan,
            ),
            0,
            1,
        )

        merged["allowed_rush_per_attempt"] = np.where(
            merged["opp_rushing_attempts"] > 0,
            merged["opp_rushing_yards"] / merged["opp_rushing_attempts"],
            np.nan,
        )
        merged["allowed_pass_per_attempt"] = np.where(
            merged["opp_passing_attempts"] > 0,
            merged["opp_passing_yards"] / merged["opp_passing_attempts"],
            np.nan,
        )
        merged["defense_success_rate"] = np.clip(
            np.where(
                merged["opp_rushing_attempts"] + merged["opp_passing_attempts"] > 0,
                1
                - (
                    (merged["opp_rushing_yards"] + merged["opp_passing_yards"]) /
                    ((merged["opp_rushing_attempts"] + merged["opp_passing_attempts"]) * 4.0)
                ),
                np.nan,
            ),
            0,
            1,
        )

        league = (
            merged.groupby(["season", "week"], as_index=False)[
                ["rush_per_attempt", "pass_per_attempt", "allowed_rush_per_attempt", "allowed_pass_per_attempt", "yards_per_play"]
            ]
            .mean()
            .rename(
                columns={
                    "rush_per_attempt": "league_rush_per_attempt",
                    "pass_per_attempt": "league_pass_per_attempt",
                    "allowed_rush_per_attempt": "league_allowed_rush_per_attempt",
                    "allowed_pass_per_attempt": "league_allowed_pass_per_attempt",
                    "yards_per_play": "league_yards_per_play",
                }
            )
        )

        merged = merged.merge(league, on=["season", "week"], how="left")
        merged["offense_rush_rating"] = (
            merged["rush_per_attempt"] - merged["league_rush_per_attempt"]
        )
        merged["offense_pass_rating"] = (
            merged["pass_per_attempt"] - merged["league_pass_per_attempt"]
        )
        merged["defense_rush_rating"] = (
            merged["league_allowed_rush_per_attempt"] - merged["allowed_rush_per_attempt"]
        )
        merged["defense_pass_rating"] = (
            merged["league_allowed_pass_per_attempt"] - merged["allowed_pass_per_attempt"]
        )

        merged["offense_epa"] = merged["offense_pass_rating"] + merged["offense_rush_rating"]
        merged["defense_epa"] = merged["defense_pass_rating"] + merged["defense_rush_rating"]
        merged["travel_penalty"] = np.nan
        merged["rest_penalty"] = np.nan
        merged["weather_adjustment"] = np.nan

        if "timezone_diff_hours" not in merged.columns:
            merged["timezone_diff_hours"] = np.nan

        result = merged[[
            "season",
            "week",
            "team",
            "offense_pass_rating",
            "offense_rush_rating",
            "defense_pass_rating",
            "defense_rush_rating",
            "pace_seconds_per_play",
            "offense_epa",
            "defense_epa",
            "offense_success_rate",
            "defense_success_rate",
            "travel_penalty",
            "rest_penalty",
            "weather_adjustment",
            "timezone_diff_hours",
        ]]

        result = result.rename(columns={"timezone_diff_hours": "avg_timezone_diff_hours"})

        if advanced_metrics is not None and not advanced_metrics.empty:
            adv_subset = advanced_metrics[[
                "season",
                "week",
                "team",
                "pace_seconds_per_play",
                "offense_epa",
                "defense_epa",
                "offense_success_rate",
                "defense_success_rate",
                "travel_penalty",
                "rest_penalty",
                "weather_adjustment",
            ]].drop_duplicates()
            if result.empty:
                result = adv_subset
                for col in [
                    "offense_pass_rating",
                    "offense_rush_rating",
                    "defense_pass_rating",
                    "defense_rush_rating",
                ]:
                    if col not in result:
                        result[col] = np.nan
            else:
                result = result.merge(
                    adv_subset,
                    on=["season", "week", "team"],
                    how="left",
                    suffixes=("", "_adv"),
                )
                for col in [
                    "pace_seconds_per_play",
                    "offense_epa",
                    "defense_epa",
                    "offense_success_rate",
                    "defense_success_rate",
                    "travel_penalty",
                    "rest_penalty",
                    "weather_adjustment",
                ]:
                    adv_col = f"{col}_adv"
                    if adv_col in result:
                        result[col] = result[col].combine_first(result[adv_col])
                        result.drop(columns=[adv_col], inplace=True)

        return result[base_columns]

    def _compute_contextual_averages(self, player_stats: pd.DataFrame) -> pd.DataFrame:
        if player_stats.empty:
            return pd.DataFrame(
                columns=
                [
                    "team",
                    "venue",
                    "day_of_week",
                    "referee",
                    "avg_rush_yards",
                    "avg_rec_yards",
                    "avg_receptions",
                    "avg_rush_tds",
                    "avg_rec_tds",
                ]
            )

        context = (
            player_stats.groupby(["team", "venue", "day_of_week", "referee"])
            .agg(
                avg_rush_yards=pd.NamedAgg(column="rushing_yards", aggfunc="mean"),
                avg_rec_yards=pd.NamedAgg(column="receiving_yards", aggfunc="mean"),
                avg_receptions=pd.NamedAgg(column="receptions", aggfunc="mean"),
                avg_rush_tds=pd.NamedAgg(column="rushing_tds", aggfunc="mean"),
                avg_rec_tds=pd.NamedAgg(column="receiving_tds", aggfunc="mean"),
            )
            .reset_index()
        )
        return context

    def _compute_team_game_rolling_stats(self, games: pd.DataFrame) -> pd.DataFrame:
        """Create rolling scoring, travel, and rest indicators for each team game."""

        base_columns = [
            "game_id",
            "season",
            "week",
            "start_time",
            "team",
            "opponent",
            "is_home",
            "rolling_points_for",
            "rolling_points_against",
            "rolling_point_diff",
            "rolling_win_pct",
            "prev_points_for",
            "prev_points_against",
            "prev_point_diff",
            "rest_days",
            "rest_penalty",
            "timezone_diff_hours",
            "travel_penalty",
        ]

        if games.empty:
            return pd.DataFrame(columns=base_columns)

        games = games.copy()
        games["start_time"] = pd.to_datetime(games["start_time"], utc=True, errors="coerce")
        games = games[games["start_time"].notna()]
        games["home_team"] = games["home_team"].apply(normalize_team_abbr)
        games["away_team"] = games["away_team"].apply(normalize_team_abbr)
        games = games.dropna(subset=["home_team", "away_team"])

        def _team_zone(team: Optional[str]) -> ZoneInfo:
            tz_name = TEAM_TIMEZONES.get(team or "", "UTC")
            try:
                return ZoneInfo(tz_name)
            except Exception:
                return ZoneInfo("UTC")

        def _tz_offset_hours(ts: dt.datetime, team: Optional[str]) -> float:
            if ts is None or pd.isna(ts):
                return 0.0
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=dt.timezone.utc)
            zone = _team_zone(team)
            offset = ts.astimezone(zone).utcoffset()
            if offset is None:
                return 0.0
            return offset.total_seconds() / 3600.0

        home = games[[
            "game_id",
            "season",
            "week",
            "start_time",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]].rename(
            columns={
                "home_team": "team",
                "away_team": "opponent",
                "home_score": "points_for",
                "away_score": "points_against",
            }
        )
        home["is_home"] = True
        home["venue_team"] = home["team"]

        away = games[[
            "game_id",
            "season",
            "week",
            "start_time",
            "away_team",
            "home_team",
            "away_score",
            "home_score",
        ]].rename(
            columns={
                "away_team": "team",
                "home_team": "opponent",
                "away_score": "points_for",
                "home_score": "points_against",
            }
        )
        away["is_home"] = False
        away["venue_team"] = away["opponent"]

        team_games = safe_concat([home, away], ignore_index=True)
        team_games = team_games.dropna(subset=["team", "opponent"])

        team_games["team"] = team_games["team"].apply(normalize_team_abbr)
        team_games["opponent"] = team_games["opponent"].apply(normalize_team_abbr)
        team_games = team_games.dropna(subset=["team", "opponent"])

        team_games["timezone_diff_hours"] = team_games.apply(
            lambda row: abs(
                _tz_offset_hours(row["start_time"], row["team"]) -
                _tz_offset_hours(row["start_time"], row.get("venue_team"))
            ),
            axis=1,
        )

        team_games = team_games.sort_values([
            "team",
            "season",
            "start_time",
            "game_id",
        ]).reset_index(drop=True)

        def compute_group(group: pd.DataFrame) -> pd.DataFrame:
            group = group.sort_values("start_time").copy()
            win_flag = np.where(
                group["points_for"].notna() & group["points_against"].notna(),
                (group["points_for"] > group["points_against"]).astype(float),
                np.nan,
            )

            group["prev_points_for"] = group["points_for"].shift(1)
            group["prev_points_against"] = group["points_against"].shift(1)
            group["prev_point_diff"] = (
                group["prev_points_for"] - group["prev_points_against"]
            )

            rolling_points_for = (
                group["points_for"].rolling(window=5, min_periods=1).mean()
            )
            rolling_points_against = (
                group["points_against"].rolling(window=5, min_periods=1).mean()
            )
            rolling_point_diff = (
                (group["points_for"] - group["points_against"]).rolling(window=5, min_periods=1).mean()
            )
            rolling_win_pct = (
                pd.Series(win_flag, index=group.index)
                .rolling(window=5, min_periods=1)
                .mean()
            )

            group["rolling_points_for"] = rolling_points_for.shift(1)
            group["rolling_points_against"] = rolling_points_against.shift(1)
            group["rolling_point_diff"] = rolling_point_diff.shift(1)
            group["rolling_win_pct"] = rolling_win_pct.shift(1)

            rest_days = group["start_time"].diff().dt.total_seconds() / 86400.0
            group["rest_days"] = rest_days
            group["rest_penalty"] = rest_days.apply(
                lambda value: max(0.0, 6.0 - value) if pd.notna(value) else np.nan
            )
            group["travel_penalty"] = np.where(
                group["is_home"],
                0.0,
                group["timezone_diff_hours"].fillna(0.0) / 3.0,
            )

            return group

        grouped_frames: List[pd.DataFrame] = []
        for _, group in team_games.groupby(["team", "season"], sort=False):
            grouped_frames.append(compute_group(group))

        if grouped_frames:
            team_games = safe_concat(grouped_frames, ignore_index=True)
        else:
            team_games = team_games.iloc[0:0]

        team_games = team_games.drop(columns=["venue_team"], errors="ignore")

        return team_games[base_columns]


# ---------------------------------------------------------------------------
# Modeling pipeline
# ---------------------------------------------------------------------------


class ModelTrainer:
    def __init__(self, engine: Engine, db: NFLDatabase, run_id: Optional[str] = None):
        self.engine = engine
        self.db = db
        self.feature_builder = FeatureBuilder(engine)
        self.run_id = run_id or uuid.uuid4().hex
        self.model_uncertainty: Dict[str, Dict[str, float]] = {}

    @staticmethod
    def _is_lineup_starter(position: str, rank: Optional[int]) -> bool:
        pos = normalize_position(position)
        if pos == "QB":
            return (rank or 99) == 1
        if pos == "RB":
            return (rank or 99) == 1
        if pos == "WR":
            return (rank or 99) in {1, 2, 3}
        if pos == "TE":
            return (rank or 99) == 1
        return False

    def apply_lineup_gate(
        self,
        player_df: pd.DataFrame,
        respect_lineups: bool = True,
        lineup_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if player_df.empty:
            return player_df

        game_ids = sorted(set(map(str, player_df["game_id"].astype(str).tolist())))
        roster_frames: List[pd.DataFrame] = []
        allowed_lineup_keys: Set[Tuple[str, str, str, str]] = set()
        lineup_audit_frame = pd.DataFrame()

        if lineup_df is not None and not lineup_df.empty:
            lineup_roster = lineup_df.copy()
            lineup_roster["game_id"] = lineup_roster["game_id"].astype(str)
            lineup_roster["team"] = lineup_roster["team"].apply(normalize_team_abbr)
            lineup_roster["position"] = lineup_roster["position"].apply(normalize_position)
            lineup_roster = lineup_roster[
                lineup_roster["position"].isin({"QB", "RB", "WR", "TE"})
            ]
            if not lineup_roster.empty:
                lineup_roster["player_id"] = lineup_roster["player_id"].fillna("").astype(str)
                lineup_roster["player_name"] = lineup_roster["player_name"].fillna("")
                if "__pname_key" not in lineup_roster.columns:
                    lineup_roster["__pname_key"] = ""
                name_seed = (
                    lineup_roster.get("first_name", "").fillna("")
                    + " "
                    + lineup_roster.get("last_name", "").fillna("")
                ).str.strip()
                fallback_name = lineup_roster["player_name"]
                lineup_roster["__pname_key"] = lineup_roster["__pname_key"].fillna("")
                need_key = lineup_roster["__pname_key"] == ""
                lineup_roster.loc[need_key, "__pname_key"] = name_seed.where(
                    name_seed != "", fallback_name
                )[need_key].map(robust_player_name_key)
                lineup_roster["__pname_key"] = lineup_roster["__pname_key"].fillna("")
                lineup_roster = lineup_roster[lineup_roster["__pname_key"] != ""].copy()
                lineup_roster["depth_rank"] = lineup_roster["rank"].apply(parse_depth_rank)
                lineup_roster["depth_rank"] = lineup_roster["depth_rank"].apply(
                    lambda val: int(val) if pd.notna(val) else None
                )
                lineup_roster["is_starter"] = lineup_roster.apply(
                    lambda row: 1
                    if self._is_lineup_starter(row["position"], row["depth_rank"])
                    else 0,
                    axis=1,
                )
                lineup_roster["source"] = "msf-lineup"
                if respect_lineups:
                    allowed_lineup_keys = {
                        (str(gid), team, name, pos)
                        for gid, team, name, pos, starter in zip(
                            lineup_roster["game_id"],
                            lineup_roster["team"],
                            lineup_roster["__pname_key"],
                            lineup_roster["position"],
                            lineup_roster["is_starter"],
                        )
                        if starter == 1 and name
                    }
                else:
                    allowed_lineup_keys = {
                        (str(gid), team, name, pos)
                        for gid, team, name, pos in zip(
                            lineup_roster["game_id"],
                            lineup_roster["team"],
                            lineup_roster["__pname_key"],
                            lineup_roster["position"],
                        )
                        if name
                    }
                lineup_audit_frame = lineup_roster.copy()
                lineup_export = lineup_roster.drop(columns=["__pname_key"], errors="ignore")
                needed_cols = [
                    "game_id",
                    "team",
                    "player_id",
                    "player_name",
                    "position",
                    "depth_rank",
                    "is_starter",
                    "source",
                ]
                for col in needed_cols:
                    if col not in lineup_export.columns:
                        lineup_export[col] = np.nan
                roster_frames.append(lineup_export[needed_cols])

        for gid in game_ids:
            roster_frame = self.db.fetch_game_roster(gid)
            if not roster_frame.empty:
                roster_frames.append(roster_frame)

        if not roster_frames:
            logging.info(
                "No roster rows found for %d games; leaving player pool unchanged",
                len(game_ids),
            )
            return player_df

        roster = safe_concat(roster_frames, ignore_index=True)
        roster["game_id"] = roster["game_id"].astype(str)

        if "source" in roster.columns:
            roster["_lineup_priority"] = roster["source"].apply(
                lambda src: 0 if str(src).startswith("msf-lineup") else 1
            )
        else:
            roster["_lineup_priority"] = 1
        roster = roster.sort_values(
            ["game_id", "team", "player_id", "player_name", "_lineup_priority"]
        )
        roster = roster.drop_duplicates(
            subset=["game_id", "team", "player_id", "player_name"], keep="first"
        )
        roster = roster.drop(columns=["_lineup_priority"], errors="ignore")

        def _nname(series: pd.Series) -> pd.Series:
            return series.fillna("").map(robust_player_name_key)

        player_df = player_df.copy()
        if "game_id" not in player_df.columns:
            player_df["game_id"] = ""
        player_df["game_id"] = player_df["game_id"].astype(str)
        player_df["player_id"] = player_df["player_id"].fillna("").astype(str)
        player_df["team"] = player_df["team"].apply(normalize_team_abbr)
        if "position" in player_df.columns:
            player_df["position"] = player_df["position"].apply(normalize_position)
        else:
            player_df["position"] = ""
        player_df["__pname_key"] = _nname(player_df["player_name"])

        roster = roster.copy()
        roster["player_id"] = roster["player_id"].fillna("").astype(str)
        roster["team"] = roster["team"].apply(normalize_team_abbr)
        roster["position"] = roster["position"].apply(normalize_position)
        roster["__pname_key"] = _nname(roster["player_name"])

        roster_subset = roster[[
            "game_id",
            "team",
            "player_id",
            "__pname_key",
            "position",
            "depth_rank",
            "is_starter",
        ]]

        merged = player_df.merge(
            roster_subset.drop(columns=["__pname_key"], errors="ignore"),
            how="left",
            left_on=["game_id", "team", "player_id"],
            right_on=["game_id", "team", "player_id"],
            suffixes=("", "_r"),
        )

        mask_missing = merged["depth_rank"].isna()
        if mask_missing.any():
            fallback = (
                merged.loc[mask_missing, ["game_id", "team", "__pname_key", "position"]]
                .merge(
                    roster_subset,
                    how="left",
                    on=["game_id", "team", "__pname_key", "position"],
                )[["depth_rank", "is_starter"]]
            )
            fallback.index = merged.index[mask_missing]
            for column in ("depth_rank", "is_starter"):
                merged.loc[fallback.index, column] = fallback[column]

        merged["_lineup_hit"] = merged["depth_rank"].notna()

        if respect_lineups and not lineup_audit_frame.empty:
            self._audit_lineup_matches(lineup_audit_frame, player_df, merged)

        candidate_pool = merged.copy()

        if respect_lineups and allowed_lineup_keys:
            key_series = pd.Series(
                list(
                    zip(
                        merged["game_id"].astype(str),
                        merged["team"],
                        merged["__pname_key"],
                        merged["position"].apply(normalize_position),
                    )
                ),
                index=merged.index,
            )
            allowed_mask = key_series.isin(allowed_lineup_keys) | merged["position"].isin(
                ["K", "DEF"]
            )
            merged = merged[allowed_mask]

        merged["depth_rank"] = merged["depth_rank"].fillna(9).astype(int)
        merged["is_starter"] = merged["is_starter"].fillna(0).astype(int)

        merged_before_filter = merged.copy()

        if respect_lineups:
            before = len(merged)
            merged = merged[
                (merged["is_starter"] == 1)
                | (merged["position"].isin(["K", "DEF"]))
            ]
            logging.info(
                "Roster gate: %d → %d players after filtering to starters (matches=%d)",
                before,
                len(merged),
                int(merged_before_filter["_lineup_hit"].sum()),
            )

            required_counts: Dict[str, int] = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
            additions: List[pd.DataFrame] = []
            starter_groups = {
                key: grp
                for key, grp in merged.groupby(["game_id", "team"], sort=False)
            }

            def _make_key(pid_value: Any, name_value: Any) -> Tuple[str, str]:
                pid_text = str(pid_value)
                if pid_text.lower() in {"", "nan", "none"}:
                    pid_text = ""
                name_text = name_value if isinstance(name_value, str) else ""
                return pid_text, name_text

            for key, full_group in candidate_pool.groupby(["game_id", "team"], sort=False):
                starter_group = starter_groups.get(key)
                if starter_group is not None:
                    existing_keys: Set[Tuple[str, str]] = {
                        _make_key(pid, name)
                        for pid, name in zip(
                            starter_group["player_id"],
                            starter_group["__pname_key"],
                        )
                    }
                    position_counts = starter_group["position"].value_counts().to_dict()
                else:
                    existing_keys = set()
                    position_counts = {}

                full_group = full_group.copy()
                if "status_bucket" in full_group.columns:
                    full_group = full_group[
                        ~full_group["status_bucket"].isin(INACTIVE_INJURY_BUCKETS)
                    ]
                if full_group.empty:
                    continue
                full_group["__fallback_key"] = [
                    _make_key(pid, name)
                    for pid, name in zip(
                        full_group["player_id"],
                        full_group["__pname_key"],
                    )
                ]

                for pos, needed in required_counts.items():
                    have = position_counts.get(pos, 0)
                    if have >= needed:
                        continue

                    candidates = full_group[full_group["position"] == pos]
                    if candidates.empty:
                        continue

                    remaining = needed - have
                    candidates = candidates[~candidates["__fallback_key"].isin(existing_keys)]
                    if candidates.empty:
                        continue

                    candidates = candidates.sort_values(["depth_rank", "player_name"])
                    selected = candidates.head(remaining)
                    if selected.empty:
                        continue

                    selected_keys: List[Tuple[str, str]] = [
                        _make_key(pid, name)
                        for pid, name in zip(
                            selected["player_id"],
                            selected["__pname_key"],
                        )
                    ]

                    additions.append(selected.drop(columns=["__fallback_key"], errors="ignore"))
                    existing_keys.update(selected_keys)
                    have += len(selected)
                    position_counts[pos] = have

                    logging.debug(
                        "Roster fallback promoted %d %s player(s) for game %s team %s",
                        len(selected_keys),
                        pos,
                        key[0],
                        key[1],
                    )

            if additions:
                merged = safe_concat([merged] + additions, ignore_index=True, sort=False)

        return merged.drop(columns=["__pname_key", "_lineup_hit"], errors="ignore")

    def _audit_lineup_matches(
        self,
        lineup_df: pd.DataFrame,
        player_df: pd.DataFrame,
        merged_df: pd.DataFrame,
    ) -> None:
        if lineup_df.empty or merged_df.empty:
            return

        try:
            lineup = lineup_df.copy()
            lineup["game_id"] = lineup["game_id"].astype(str)
            lineup["team"] = lineup["team"].apply(normalize_team_abbr)
            lineup["position"] = lineup["position"].apply(normalize_position)
            lineup = lineup[lineup["position"].isin({"QB", "RB", "WR", "TE"})]
            if lineup.empty:
                return

            if "__pname_key" not in lineup.columns:
                lineup["__pname_key"] = ""
            name_seed = (
                lineup.get("first_name", "").fillna("")
                + " "
                + lineup.get("last_name", "").fillna("")
            ).str.strip()
            fallback = lineup.get("player_name", "").fillna("")
            need_key = lineup["__pname_key"].fillna("") == ""
            lineup.loc[need_key, "__pname_key"] = name_seed.where(
                name_seed != "", fallback
            )[need_key].map(robust_player_name_key)
            lineup["__pname_key"] = lineup["__pname_key"].fillna("")
            lineup = lineup[lineup["__pname_key"] != ""]
            if lineup.empty:
                return

            players = player_df.copy()
            players["game_id"] = players["game_id"].astype(str)
            players["team"] = players["team"].apply(normalize_team_abbr)
            players["position"] = players["position"].apply(normalize_position)
            if "__pname_key" not in players.columns:
                players["__pname_key"] = players["player_name"].map(
                    robust_player_name_key
                )

            matched_keys = set(
                zip(
                    merged_df.loc[merged_df["_lineup_hit"], "game_id"].astype(str),
                    merged_df.loc[merged_df["_lineup_hit"], "team"],
                    merged_df.loc[merged_df["_lineup_hit"], "__pname_key"],
                    merged_df.loc[merged_df["_lineup_hit"], "position"].apply(
                        normalize_position
                    ),
                )
            )

            reported: Set[Tuple[str, str, str]] = set()
            for row in lineup.itertuples():
                key = (row.game_id, row.team, row.__pname_key, row.position)
                if key in matched_keys:
                    continue
                summary_key = (row.game_id, row.team, row.__pname_key)
                if summary_key in reported:
                    continue
                team_pool = players[
                    (players["game_id"] == row.game_id)
                    & (players["team"] == row.team)
                ]
                reasons: List[str] = []
                if team_pool.empty:
                    reasons.append("team missing in features")
                else:
                    name_pool = team_pool[team_pool["__pname_key"] == row.__pname_key]
                    if name_pool.empty:
                        reasons.append("not in latest_players")
                    else:
                        pos_pool = name_pool[
                            name_pool["position"].apply(normalize_position)
                            == row.position
                        ]
                        if pos_pool.empty:
                            reasons.append("position mismatch")
                        else:
                            inactive_mask = pd.Series(False, index=pos_pool.index)
                            if "status_bucket" in pos_pool.columns:
                                inactive_mask = pos_pool["status_bucket"].isin(
                                    INACTIVE_INJURY_BUCKETS
                                )
                                if inactive_mask.any():
                                    reasons.append("inactive status filtered")
                            if not inactive_mask.any():
                                reasons.append("present but filtered")

                if not reasons:
                    reasons.append("unmatched")

                player_label = getattr(row, "player_name", "").strip() or (
                    " ".join(
                        part
                        for part in [
                            getattr(row, "first_name", ""),
                            getattr(row, "last_name", ""),
                        ]
                        if part
                    )
                )
                logging.warning(
                    "[%s %s %s-%s] %s: %s",
                    row.game_id,
                    row.team,
                    row.position,
                    getattr(row, "rank", ""),
                    player_label,
                    ", ".join(reasons),
                )
                reported.add(summary_key)
        except Exception:
            logging.debug("Lineup audit diagnostics failed", exc_info=True)

    # ------------------------------------------------------------------
    # Chronological splitting utilities
    # ------------------------------------------------------------------

    def _sort_by_time(self, df: pd.DataFrame) -> pd.DataFrame:
        if "start_time" in df.columns:
            return df.sort_values("start_time")
        if {"season", "week"}.issubset(df.columns):
            return df.sort_values(["season", "week"])
        if "week" in df.columns:
            return df.sort_values("week")
        return df.sort_index()

    def _chronological_split(
        self,
        df: pd.DataFrame,
        holdout_fraction: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_sorted = self._sort_by_time(df).reset_index(drop=True)
        if len(df_sorted) < 5:
            split_index = max(1, len(df_sorted) - 1)
        else:
            holdout_size = max(1, int(len(df_sorted) * holdout_fraction))
            if holdout_size >= len(df_sorted):
                holdout_size = max(1, len(df_sorted) - 1)
            split_index = len(df_sorted) - holdout_size

        if split_index <= 0 or split_index >= len(df_sorted):
            split_index = max(1, len(df_sorted) - 1)

        train_df = df_sorted.iloc[:split_index]
        test_df = df_sorted.iloc[split_index:]
        return train_df, test_df, df_sorted

    def _build_time_series_cv(self, n_samples: int) -> TimeSeriesSplit:
        if n_samples < 3:
            raise ValueError("At least 3 samples are required for time series CV.")

        n_splits = min(5, max(2, n_samples - 1))
        if n_splits >= n_samples:
            n_splits = n_samples - 1
        return TimeSeriesSplit(n_splits=n_splits)

    @staticmethod
    def _gb_param_grid(prefix: str) -> Dict[str, List[Any]]:
        """Gradient boosting hyperparameter search space helper."""

        return {
            f"{prefix}learning_rate": [0.01, 0.05, 0.1, 0.2],
            f"{prefix}n_estimators": [100, 200, 300, 400],
            f"{prefix}max_depth": [2, 3, 4],
            f"{prefix}subsample": [0.6, 0.8, 1.0],
        }

    def train(self) -> Dict[str, Pipeline]:
        datasets = self.feature_builder.build_features()
        models: Dict[str, Pipeline] = {}

        for target, df in datasets.items():
            if target == "game_outcome":
                model = self._train_game_models(df)
                models.update(model)
                continue

            model = self._train_regression_model(df, target)
            if model is not None:
                models[target] = model
        return models

    def _train_regression_model(self, df: pd.DataFrame, target: str) -> Optional[Pipeline]:
        if len(df) < 20 or df[target].nunique() <= 1:
            logging.warning(
                "Not enough data to train %s model (rows=%d, unique targets=%d).", 
                target,
                len(df),
                df[target].nunique(),
            )
            return None

        numeric_features = [
            "week",
            "temperature_f",
            "wind_mph",
            "humidity",
            "offense_pass_rating",
            "offense_rush_rating",
            "defense_pass_rating",
            "defense_rush_rating",
            "pace_seconds_per_play",
            "offense_epa",
            "defense_epa",
            "offense_success_rate",
            "defense_success_rate",
            "travel_penalty",
            "rest_penalty",
            "weather_adjustment",
            "avg_timezone_diff_hours",
            "opp_offense_pass_rating",
            "opp_offense_rush_rating",
            "opp_defense_pass_rating",
            "opp_defense_rush_rating",
            "opp_pace_seconds_per_play",
            "opp_offense_epa",
            "opp_defense_epa",
            "opp_offense_success_rate",
            "opp_defense_success_rate",
            "opp_travel_penalty",
            "opp_rest_penalty",
            "opp_weather_adjustment",
            "opp_timezone_diff_hours",
            "avg_rush_yards",
            "avg_rec_yards",
            "avg_receptions",
            "avg_rush_tds",
            "avg_rec_tds",
            "snap_count",
            "receiving_targets",
            "home_injury_total",
            "away_injury_total",
            "depth_rank",
            "injury_priority",
            "practice_priority",
        ]
        categorical_features = [
            "team",
            "opponent",
            "venue",
            "day_of_week",
            "referee",
            "position",
            "status_bucket",
            "practice_status",
        ]

        available_numeric = [
            col for col in numeric_features if col in df.columns and df[col].notna().any()
        ]
        dropped_numeric = sorted(set(numeric_features) - set(available_numeric))
        if dropped_numeric:
            logging.debug(
                "Dropping numeric features with no observed values for %s model: %s",
                target,
                ", ".join(dropped_numeric),
            )

        available_categorical = [
            col for col in categorical_features if col in df.columns and df[col].notna().any()
        ]
        dropped_categorical = sorted(set(categorical_features) - set(available_categorical))
        if dropped_categorical:
            logging.debug(
                "Dropping categorical features with no observed values for %s model: %s",
                target,
                ", ".join(dropped_categorical),
            )

        if not available_numeric and not available_categorical:
            logging.warning(
                "No usable features with observed values available to train %s model; skipping.",
                target,
            )
            return None

        feature_columns = list(available_numeric + available_categorical)

        train_df, test_df, sorted_df = self._chronological_split(df)
        X_train = train_df[feature_columns]
        y_train = train_df[target]
        X_test = test_df[feature_columns]
        y_test = test_df[target]

        transformers = []
        if available_numeric:
            transformers.append(
                (
                    "num",
                    Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]),
                    available_numeric,
                )
            )
        if available_categorical:
            transformers.append(
                (
                    "cat",
                    Pipeline([
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="missing"),
                        ),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]),
                    available_categorical,
                )
            )

        preprocessor = ColumnTransformer(transformers=transformers)

        baseline_model = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("regressor", GradientBoostingRegressor(random_state=42)),
        ])

        baseline_model.fit(X_train, y_train)
        baseline_pred = baseline_model.predict(X_test)
        baseline_r2 = baseline_model.score(X_test, y_test)
        baseline_mae = mean_absolute_error(y_test, baseline_pred)
        baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))
        logging.info(
            "Trained %s model (baseline GBM), R^2=%.3f on holdout (MAE=%.3f, RMSE=%.3f)",
            target,
            baseline_r2,
            baseline_mae,
            baseline_rmse,
        )
        self.db.record_backtest_metrics(
            self.run_id,
            f"{target}_baseline",
            {"r2": baseline_r2, "mae": baseline_mae, "rmse": baseline_rmse},
            sample_size=len(y_test),
        )

        tuned_model = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("regressor", GradientBoostingRegressor(random_state=42)),
        ])

        try:
            cv = self._build_time_series_cv(len(X_train))
        except ValueError as exc:
            logging.warning(
                "Skipping hyperparameter tuning for %s due to insufficient data: %s",
                target,
                exc,
            )
            best_model = tuned_model.fit(X_train, y_train)
        else:
            search = RandomizedSearchCV(
                estimator=tuned_model,
                param_distributions=self._gb_param_grid("regressor__"),
                n_iter=10,
                scoring="neg_mean_absolute_error",
                cv=cv,
                random_state=42,
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            best_model: Pipeline = search.best_estimator_
            logging.info(
                "Best parameters for %s model: %s (CV MAE=%.3f)",
                target,
                search.best_params_,
                -search.best_score_,
            )

        rf_pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=400, random_state=42, min_samples_leaf=2, n_jobs=-1
                ),
            ),
        ])

        final_estimator = GradientBoostingRegressor(
            random_state=42, learning_rate=0.05, max_depth=3, n_estimators=200
        )

        ensemble = StackingRegressor(
            estimators=[
                ("gbm", clone(best_model)),
                ("rf", rf_pipeline),
            ],
            final_estimator=final_estimator,
            passthrough=False,
            n_jobs=-1,
        )

        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        r2 = ensemble.score(X_test, y_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        logging.info(
            "%s holdout metrics | R^2=%.3f | MAE=%.3f | RMSE=%.3f",
            target,
            r2,
            mae,
            rmse,
        )

        self.db.record_backtest_metrics(
            self.run_id,
            target,
            {"r2": r2, "mae": mae, "rmse": rmse},
            sample_size=len(y_test),
        )
        self.model_uncertainty[target] = {"rmse": rmse, "mae": mae}

        ensemble.fit(sorted_df[feature_columns], sorted_df[target])
        setattr(ensemble, "feature_columns", feature_columns)
        setattr(ensemble, "allowed_positions", TARGET_ALLOWED_POSITIONS.get(target))
        setattr(ensemble, "target_name", target)
        return ensemble


    def _train_game_models(self, df: pd.DataFrame) -> Dict[str, Pipeline]:
        if len(df) < 20 or df["game_result"].nunique() <= 1:
            logging.warning(
                "Not enough completed games (%d) with outcomes to train game-level models.",
                len(df),
            )
            return {}

        df = df.copy()

        numeric_features = [
            "week",
            "temperature_f",
            "wind_mph",
            "humidity",
            "home_moneyline",
            "away_moneyline",
            "home_implied_prob",
            "away_implied_prob",
            "moneyline_diff",
            "implied_prob_diff",
            "implied_prob_sum",
            "home_offense_pass_rating",
            "home_offense_rush_rating",
            "home_defense_pass_rating",
            "home_defense_rush_rating",
            "home_pace_seconds_per_play",
            "home_offense_epa",
            "home_defense_epa",
            "home_offense_success_rate",
            "home_defense_success_rate",
            "home_travel_penalty",
            "home_rest_penalty",
            "home_weather_adjustment",
            "home_timezone_diff_hours",
            "home_points_for_avg",
            "home_points_against_avg",
            "home_point_diff_avg",
            "home_win_pct_recent",
            "home_prev_points_for",
            "home_prev_points_against",
            "home_prev_point_diff",
            "home_rest_days",
            "away_offense_pass_rating",
            "away_offense_rush_rating",
            "away_defense_pass_rating",
            "away_defense_rush_rating",
            "away_pace_seconds_per_play",
            "away_offense_epa",
            "away_defense_epa",
            "away_offense_success_rate",
            "away_defense_success_rate",
            "away_travel_penalty",
            "away_rest_penalty",
            "away_weather_adjustment",
            "away_timezone_diff_hours",
            "away_points_for_avg",
            "away_points_against_avg",
            "away_point_diff_avg",
            "away_win_pct_recent",
            "away_prev_points_for",
            "away_prev_points_against",
            "away_prev_point_diff",
            "away_rest_days",
        ]

        injury_columns = [
            col
            for col in df.columns
            if col.startswith("home_injury_") or col.startswith("away_injury_")
        ]
        numeric_features.extend(sorted(injury_columns))

        categorical_features = [
            "venue",
            "day_of_week",
            "referee",
            "weather_conditions",
            "home_team",
            "away_team",
        ]

        available_numeric = [
            col for col in numeric_features if col in df.columns and df[col].notna().any()
        ]
        dropped_numeric = sorted(set(numeric_features) - set(available_numeric))
        if dropped_numeric:
            logging.debug(
                "Dropping numeric game features with no observed values: %s",
                ", ".join(dropped_numeric),
            )

        available_categorical = [
            col for col in categorical_features if col in df.columns and df[col].notna().any()
        ]
        dropped_categorical = sorted(set(categorical_features) - set(available_categorical))
        if dropped_categorical:
            logging.debug(
                "Dropping categorical game features with no observed values: %s",
                ", ".join(dropped_categorical),
            )

        if not available_numeric and not available_categorical:
            logging.warning(
                "No usable features with observed values available to train game-level models.",
            )
            return {}

        feature_columns = available_numeric + available_categorical

        train_df, test_df, sorted_df = self._chronological_split(df)
        X_train = train_df[feature_columns]
        X_test = test_df[feature_columns]

        y_winner_train = (train_df["game_result"] == "home").astype(int)
        y_winner_test = (test_df["game_result"] == "home").astype(int)

        y_home_train = train_df["home_score"]
        y_home_test = test_df["home_score"]

        y_away_train = train_df["away_score"]
        y_away_test = test_df["away_score"]

        transformers = []
        if available_numeric:
            transformers.append(
                (
                    "num",
                    Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())]),
                    available_numeric,
                )
            )
        if available_categorical:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(strategy="constant", fill_value="missing"),
                            ),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    available_categorical,
                )
            )

        preprocessor = ColumnTransformer(transformers=transformers)

        baseline_clf = Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                ("classifier", GradientBoostingClassifier(random_state=42)),
            ]
        )
        baseline_home = Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                ("regressor", GradientBoostingRegressor(random_state=42)),
            ]
        )
        baseline_away = Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                ("regressor", GradientBoostingRegressor(random_state=42)),
            ]
        )

        baseline_clf.fit(X_train, y_winner_train)
        baseline_home.fit(X_train, y_home_train)
        baseline_away.fit(X_train, y_away_train)

        logging.info(
            "Trained game outcome classifier (baseline), accuracy=%.3f",
            baseline_clf.score(X_test, y_winner_test),
        )
        logging.info(
            "Trained home score regressor (baseline), R^2=%.3f",
            baseline_home.score(X_test, y_home_test),
        )
        logging.info(
            "Trained away score regressor (baseline), R^2=%.3f",
            baseline_away.score(X_test, y_away_test),
        )

        tuned_clf = Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                ("classifier", GradientBoostingClassifier(random_state=42)),
            ]
        )
        tuned_home = Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                ("regressor", GradientBoostingRegressor(random_state=42)),
            ]
        )
        tuned_away = Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                ("regressor", GradientBoostingRegressor(random_state=42)),
            ]
        )

        try:
            cv = self._build_time_series_cv(len(X_train))
        except ValueError as exc:
            logging.warning(
                "Skipping hyperparameter tuning for game models due to insufficient data: %s",
                exc,
            )
            best_clf = tuned_clf.fit(X_train, y_winner_train)
            best_reg_home = tuned_home.fit(X_train, y_home_train)
            best_reg_away = tuned_away.fit(X_train, y_away_train)
        else:
            clf_search = RandomizedSearchCV(
                estimator=tuned_clf,
                param_distributions=self._gb_param_grid("classifier__"),
                n_iter=10,
                scoring="roc_auc",
                cv=cv,
                random_state=42,
                n_jobs=-1,
            )
            clf_search.fit(X_train, y_winner_train)
            best_clf = clf_search.best_estimator_
            logging.info(
                "Best parameters for game winner model: %s (CV ROC-AUC=%.3f)",
                clf_search.best_params_,
                clf_search.best_score_,
            )

            home_search = RandomizedSearchCV(
                estimator=tuned_home,
                param_distributions=self._gb_param_grid("regressor__"),
                n_iter=10,
                scoring="neg_mean_absolute_error",
                cv=cv,
                random_state=42,
                n_jobs=-1,
            )
            home_search.fit(X_train, y_home_train)
            best_reg_home = home_search.best_estimator_
            logging.info(
                "Best parameters for home score model: %s (CV MAE=%.3f)",
                home_search.best_params_,
                -home_search.best_score_,
            )

            away_search = RandomizedSearchCV(
                estimator=tuned_away,
                param_distributions=self._gb_param_grid("regressor__"),
                n_iter=10,
                scoring="neg_mean_absolute_error",
                cv=cv,
                random_state=42,
                n_jobs=-1,
            )
            away_search.fit(X_train, y_away_train)
            best_reg_away = away_search.best_estimator_
            logging.info(
                "Best parameters for away score model: %s (CV MAE=%.3f)",
                away_search.best_params_,
                -away_search.best_score_,
            )

        rf_clf = Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=500,
                        random_state=42,
                        min_samples_leaf=2,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        stack_clf = StackingClassifier(
            estimators=[("gbm", clone(best_clf)), ("rf", rf_clf)],
            final_estimator=LogisticRegression(max_iter=1000),
            passthrough=False,
            n_jobs=-1,
        )

        try:
            calibrated_clf: Pipeline = CalibratedClassifierCV(
                estimator=stack_clf,
                method="sigmoid",
                cv=min(3, max(2, len(np.unique(y_winner_train)))),
            )
            calibrated_clf.fit(X_train, y_winner_train)
            final_clf: Pipeline = calibrated_clf
        except ValueError as exc:
            logging.warning("Calibration skipped for game winner model: %s", exc)
            final_clf = stack_clf.fit(X_train, y_winner_train)

        rf_home = Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=600,
                        random_state=42,
                        min_samples_leaf=2,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        final_home = StackingRegressor(
            estimators=[("gbm", clone(best_reg_home)), ("rf", rf_home)],
            final_estimator=GradientBoostingRegressor(
                random_state=42, learning_rate=0.05, max_depth=3, n_estimators=200
            ),
            passthrough=False,
            n_jobs=-1,
        )
        final_home.fit(X_train, y_home_train)

        rf_away = Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=600,
                        random_state=42,
                        min_samples_leaf=2,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        final_away = StackingRegressor(
            estimators=[("gbm", clone(best_reg_away)), ("rf", rf_away)],
            final_estimator=GradientBoostingRegressor(
                random_state=42, learning_rate=0.05, max_depth=3, n_estimators=200
            ),
            passthrough=False,
            n_jobs=-1,
        )
        final_away.fit(X_train, y_away_train)

        if hasattr(final_clf, "predict_proba"):
            winner_proba = final_clf.predict_proba(X_test)[:, 1]
        else:
            decision = final_clf.decision_function(X_test)
            winner_proba = 1.0 / (1.0 + np.exp(-decision))
        winner_pred = (winner_proba >= 0.5).astype(int)
        winner_accuracy = accuracy_score(y_winner_test, winner_pred)
        try:
            winner_roc_auc = roc_auc_score(y_winner_test, winner_proba)
        except ValueError:
            winner_roc_auc = float("nan")
        try:
            winner_log_loss = log_loss(y_winner_test, winner_proba, labels=[0, 1])
        except ValueError:
            winner_log_loss = float("nan")
        try:
            winner_brier = brier_score_loss(y_winner_test, winner_proba)
        except ValueError:
            winner_brier = float("nan")

        logging.info(
            "Game winner holdout metrics | accuracy=%.3f | ROC-AUC=%s | log_loss=%s | brier=%s",
            winner_accuracy,
            f"{winner_roc_auc:.3f}" if not np.isnan(winner_roc_auc) else "nan",
            f"{winner_log_loss:.3f}" if not np.isnan(winner_log_loss) else "nan",
            f"{winner_brier:.3f}" if not np.isnan(winner_brier) else "nan",
        )

        home_pred = final_home.predict(X_test)
        home_r2 = final_home.score(X_test, y_home_test)
        home_mae = mean_absolute_error(y_home_test, home_pred)
        home_rmse = float(np.sqrt(mean_squared_error(y_home_test, home_pred)))
        logging.info(
            "Home score holdout metrics | R^2=%.3f | MAE=%.3f | RMSE=%.3f",
            home_r2,
            home_mae,
            home_rmse,
        )

        away_pred = final_away.predict(X_test)
        away_r2 = final_away.score(X_test, y_away_test)
        away_mae = mean_absolute_error(y_away_test, away_pred)
        away_rmse = float(np.sqrt(mean_squared_error(y_away_test, away_pred)))
        logging.info(
            "Away score holdout metrics | R^2=%.3f | MAE=%.3f | RMSE=%.3f",
            away_r2,
            away_mae,
            away_rmse,
        )

        self.db.record_backtest_metrics(
            self.run_id,
            "game_winner",
            {
                "accuracy": winner_accuracy,
                "roc_auc": winner_roc_auc,
                "log_loss": winner_log_loss,
                "brier": winner_brier,
            },
            sample_size=len(y_winner_test),
        )
        self.db.record_backtest_metrics(
            self.run_id,
            "home_points",
            {"r2": home_r2, "mae": home_mae, "rmse": home_rmse},
            sample_size=len(y_home_test),
        )
        self.db.record_backtest_metrics(
            self.run_id,
            "away_points",
            {"r2": away_r2, "mae": away_mae, "rmse": away_rmse},
            sample_size=len(y_away_test),
        )

        self.model_uncertainty["game_winner"] = {
            "log_loss": winner_log_loss,
            "brier": winner_brier,
            "accuracy": winner_accuracy,
        }
        self.model_uncertainty["home_points"] = {"rmse": home_rmse, "mae": home_mae}
        self.model_uncertainty["away_points"] = {"rmse": away_rmse, "mae": away_mae}

        X_full = sorted_df[feature_columns]
        y_winner_full = (sorted_df["game_result"] == "home").astype(int)
        final_clf.fit(X_full, y_winner_full)
        final_home.fit(X_full, sorted_df["home_score"])
        final_away.fit(X_full, sorted_df["away_score"])

        setattr(final_clf, "feature_columns", feature_columns)
        setattr(final_home, "feature_columns", feature_columns)
        setattr(final_away, "feature_columns", feature_columns)

        return {
            "game_winner": final_clf,
            "home_points": final_home,
            "away_points": final_away,
        }


# ---------------------------------------------------------------------------
# Prediction utilities
# ---------------------------------------------------------------------------


def predict_upcoming_games(
    models: Dict[str, Pipeline],
    engine: Engine,
    model_uncertainty: Optional[Dict[str, Dict[str, float]]] = None,
    output_path: Optional[Path] = None,
    save_json: bool = False,
    ingestor: Optional["NFLIngestor"] = None,
    trainer: Optional["ModelTrainer"] = None,
    config: Optional[NFLConfig] = None,
) -> Dict[str, pd.DataFrame]:
    feature_builder = FeatureBuilder(engine)
    feature_builder.build_features()
    model_uncertainty = model_uncertainty or {}

    base_games = pd.read_sql_table("nfl_games", engine).rename(columns=lambda col: str(col))
    base_games["start_time"] = pd.to_datetime(base_games["start_time"])
    base_games["day_of_week"] = base_games["day_of_week"].where(
        base_games["day_of_week"].notna(), base_games["start_time"].dt.day_name()
    )

    games_source = feature_builder.games_frame
    if games_source is not None and not games_source.empty:
        games_source = games_source.copy()
        games_source = games_source.rename(columns=lambda col: str(col))
    else:
        games_source = base_games

    games_source["start_time"] = pd.to_datetime(games_source["start_time"])
    games_source["day_of_week"] = games_source["day_of_week"].where(
        games_source["day_of_week"].notna(), games_source["start_time"].dt.day_name()
    )

    if "odds_updated" not in games_source.columns:
        games_source["odds_updated"] = pd.NaT

    upcoming_mask = (games_source["status"].isin(["upcoming", "scheduled", "inprogress"])) | games_source["home_score"].isna()
    upcoming = games_source.loc[upcoming_mask].copy()
    if upcoming.empty:
        logging.warning("No upcoming games found for prediction")
        return {"games": pd.DataFrame(), "players": pd.DataFrame()}

    upcoming["home_team"] = upcoming["home_team"].apply(normalize_team_abbr)
    upcoming["away_team"] = upcoming["away_team"].apply(normalize_team_abbr)
    upcoming = upcoming[upcoming["home_team"].notna() & upcoming["away_team"].notna()]
    if upcoming.empty:
        logging.warning("Upcoming games are missing team assignments after normalization")
        return {"games": pd.DataFrame(), "players": pd.DataFrame()}

    upcoming["start_time"] = pd.to_datetime(upcoming["start_time"], utc=True, errors="coerce")
    upcoming = upcoming[upcoming["start_time"].notna()]
    if upcoming.empty:
        logging.warning("Upcoming games are missing valid start times after normalization")
        return {"games": pd.DataFrame(), "players": pd.DataFrame()}

    eastern = ZoneInfo("America/New_York")
    upcoming["local_start_time"] = upcoming["start_time"].dt.tz_convert(eastern)
    upcoming["local_day_of_week"] = upcoming["local_start_time"].dt.day_name()
    upcoming["day_of_week"] = upcoming["day_of_week"].where(
        upcoming["day_of_week"].notna(), upcoming["local_day_of_week"]
    )

    now_utc = dt.datetime.now(dt.timezone.utc)
    lookback = now_utc - pd.Timedelta(hours=12)
    lookahead = now_utc + pd.Timedelta(days=7, hours=12)
    in_window_mask = (upcoming["start_time"] >= lookback) & (
        upcoming["start_time"] <= lookahead
    )
    window_games = upcoming.loc[in_window_mask].copy()

    if window_games.empty:
        earliest_start = upcoming["start_time"].min()
        if pd.isna(earliest_start):
            logging.warning("No upcoming games have a valid kickoff time available")
            return {"games": pd.DataFrame(), "players": pd.DataFrame()}
        week_start = earliest_start.normalize() - pd.to_timedelta(earliest_start.weekday(), unit="D")
        week_end = week_start + pd.Timedelta(days=7)
        week_mask = (upcoming["start_time"] >= week_start) & (
            upcoming["start_time"] <= week_end
        )
        window_games = upcoming.loc[week_mask].copy()
        if window_games.empty:
            logging.warning("No upcoming games within the current week window")
            return {"games": pd.DataFrame(), "players": pd.DataFrame()}
        logging.info(
            "Falling back to earliest scheduled week %s-%s with %d games",
            week_start.date(),
            week_end.date(),
            len(window_games),
        )

    upcoming = window_games.copy()

    desired_days = {"Thursday", "Sunday", "Monday"}
    upcoming = upcoming[upcoming["local_day_of_week"].isin(desired_days)]
    if upcoming.empty:
        logging.warning("No Thursday/Sunday/Monday games available for prediction")
        return {"games": pd.DataFrame(), "players": pd.DataFrame()}

    upcoming.loc[:, "_priority"] = upcoming["game_id"].apply(
        lambda value: 0 if isinstance(value, str) and value.isdigit() else 1
    )
    upcoming = upcoming.sort_values(
        ["_priority", "odds_updated", "start_time"], ascending=[True, False, True]
    )
    upcoming = upcoming.drop_duplicates(
        subset=["home_team", "away_team", "start_time"], keep="first"
    )
    upcoming = upcoming.drop(columns="_priority", errors="ignore")

    upcoming = upcoming.sort_values("start_time").reset_index(drop=True)

    def _ensure_model_features(frame: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
        columns = getattr(model, "feature_columns", None)
        if not columns:
            return frame
        missing = [col for col in columns if col not in frame.columns]
        if missing:
            frame = frame.copy()
            for col in missing:
                frame[col] = np.nan
        return frame[columns]

    # Game-level predictions
    game_models_present = all(key in models for key in ("game_winner", "home_points", "away_points"))
    if not game_models_present:
        logging.error("Missing trained game-level models. Cannot generate scoreboard predictions.")
        return {"games": pd.DataFrame(), "players": pd.DataFrame()}

    game_features = feature_builder.prepare_upcoming_game_features(upcoming)
    home_features = _ensure_model_features(game_features, models["home_points"])
    away_features = _ensure_model_features(game_features, models["away_points"])
    winner_features = _ensure_model_features(game_features, models["game_winner"])

    away_predictions = models["away_points"].predict(away_features)
    home_predictions = models["home_points"].predict(home_features)
    winner_probs = models["game_winner"].predict_proba(winner_features)[:, 1]

    scoreboard = upcoming[[
        "game_id",
        "start_time",
        "local_start_time",
        "away_team",
        "home_team",
    ]].copy()
    scoreboard["away_score"] = away_predictions
    scoreboard["home_score"] = home_predictions
    scoreboard["home_win_probability"] = winner_probs

    home_unc = (model_uncertainty.get("home_points") or {})
    away_unc = (model_uncertainty.get("away_points") or {})
    winner_unc = (model_uncertainty.get("game_winner") or {})

    home_rmse = float(home_unc.get("rmse")) if home_unc.get("rmse") is not None else np.nan
    away_rmse = float(away_unc.get("rmse")) if away_unc.get("rmse") is not None else np.nan

    def _interval_bounds(series: pd.Series, rmse: float) -> Tuple[pd.Series, pd.Series]:
        if pd.isna(rmse):
            return pd.Series(np.nan, index=series.index), pd.Series(np.nan, index=series.index)
        lower = series - rmse
        upper = series + rmse
        return lower.clip(lower=0.0), upper.clip(lower=0.0)

    home_lower, home_upper = _interval_bounds(scoreboard["home_score"], home_rmse)
    away_lower, away_upper = _interval_bounds(scoreboard["away_score"], away_rmse)
    scoreboard["home_score_lower"] = home_lower
    scoreboard["home_score_upper"] = home_upper
    scoreboard["away_score_lower"] = away_lower
    scoreboard["away_score_upper"] = away_upper

    scoreboard["home_win_log_loss"] = winner_unc.get("log_loss")
    scoreboard["home_win_brier"] = winner_unc.get("brier")
    scoreboard["home_win_accuracy"] = winner_unc.get("accuracy")
    scoreboard["date"] = scoreboard["local_start_time"].dt.date.astype(str)
    scoreboard = scoreboard[
        [
            "game_id",
            "date",
            "start_time",
            "local_start_time",
            "away_team",
            "home_team",
            "away_score",
            "home_score",
            "away_score_lower",
            "away_score_upper",
            "home_score_lower",
            "home_score_upper",
            "home_win_probability",
            "home_win_log_loss",
            "home_win_brier",
            "home_win_accuracy",
        ]
    ].rename(
        columns={
            "away_team": "away_team_abbr",
            "home_team": "home_team_abbr",
        }
    )
    scoreboard = scoreboard.sort_values(["date", "start_time", "game_id"]).reset_index(drop=True)

    # Player-level predictions
    lineup_df = pd.DataFrame()
    if ingestor is not None:
        lineup_cache: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
        lineup_records: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        for game in upcoming.itertuples():
            season_val = getattr(game, "season", None)
            start_time = getattr(game, "start_time", None)
            away_team = getattr(game, "away_team", None)
            home_team = getattr(game, "home_team", None)
            if not away_team or not home_team:
                continue
            rows = ingestor.fetch_lineup_rows(
                start_time,
                away_team,
                home_team,
                cache=lineup_cache,
            )
            for row in rows:
                team = normalize_team_abbr(row.get("team"))
                position = normalize_position(row.get("position"))
                if not team or not position:
                    continue
                player_name = row.get("player_name") or ""
                first_name = row.get("first_name") or ""
                last_name = row.get("last_name") or ""
                name_for_key = " ".join(part for part in [first_name, last_name] if part) or player_name
                pname_key = row.get("__pname_key") or robust_player_name_key(name_for_key)
                if not player_name or not pname_key:
                    continue
                player_name_norm = normalize_player_name(player_name)
                raw_player_id = row.get("player_id")
                player_id = (
                    str(raw_player_id)
                    if raw_player_id not in (None, "")
                    else f"lineup_{team}_{pname_key}"
                )
                updated_at = row.get("updated_at")
                if isinstance(updated_at, str):
                    updated_at = parse_dt(updated_at)
                depth_id = row.get("depth_id")
                if not depth_id:
                    depth_id = f"msf-lineup:{team}:{position}:{pname_key}"

                status_bucket = row.get("status_bucket")
                practice_status = row.get("practice_status")
                if status_bucket:
                    status_bucket = normalize_injury_status(status_bucket)
                    practice_status = normalize_practice_status(practice_status)
                else:
                    status_bucket, practice_status = interpret_playing_probability(
                        row.get("playing_probability")
                    )
                    status_bucket = normalize_injury_status(status_bucket)
                    practice_status = normalize_practice_status(practice_status)

                record = {
                    "game_id": str(getattr(game, "game_id", "")),
                    "depth_id": depth_id,
                    "team": team,
                    "position": position,
                    "player_id": player_id,
                    "player_name": player_name,
                    "first_name": first_name,
                    "last_name": last_name,
                    "rank": row.get("rank"),
                    "updated_at": updated_at,
                    "player_name_norm": player_name_norm,
                    "__pname_key": pname_key,
                    "game_start": row.get("game_start") or start_time,
                    "side": row.get("side"),
                    "base_pos": row.get("base_pos") or position,
                    "playing_probability": row.get("playing_probability"),
                    "player_team": row.get("player_team"),
                    "status_bucket": status_bucket,
                    "practice_status": practice_status,
                }
                key = (record["game_id"], team, pname_key, position)
                existing = lineup_records.get(key)
                existing_ts = existing.get("updated_at") if existing else None
                existing_ts = pd.to_datetime(existing_ts) if existing_ts is not None else None
                new_ts = pd.to_datetime(updated_at) if updated_at is not None else None
                if existing is None or (existing_ts or pd.Timestamp.min) <= (new_ts or pd.Timestamp.max):
                    lineup_records[key] = record

        if lineup_records:
            lineup_df = pd.DataFrame(lineup_records.values())

    player_features = feature_builder.prepare_upcoming_player_features(
        upcoming,
        lineup_rows=lineup_df if not lineup_df.empty else None,
    )
    respect_lineups = True if config is None else bool(config.respect_lineups)
    if trainer is not None:
        player_features = trainer.apply_lineup_gate(
            player_features,
            respect_lineups=respect_lineups,
            lineup_df=lineup_df if not lineup_df.empty else None,
        )
    elif respect_lineups:
        logging.warning(
            "Respect-lineups flag enabled but no trainer provided; skipping roster gating",
        )
    player_predictions = pd.DataFrame()
    if not player_features.empty:
        player_predictions = player_features[
            ["game_id", "team", "player_id", "player_name", "position"]
        ].copy()

        for target, model in models.items():
            if target in {"game_winner", "home_points", "away_points"}:
                continue

            allowed_positions = getattr(
                model, "allowed_positions", TARGET_ALLOWED_POSITIONS.get(target)
            )
            if allowed_positions:
                mask = player_predictions["position"].isin(allowed_positions)
            else:
                mask = pd.Series(True, index=player_predictions.index)

            target_values = pd.Series(np.nan, index=player_predictions.index, dtype=float)
            if mask.any():
                features_for_model = _ensure_model_features(
                    player_features.loc[mask], model
                )
                try:
                    preds = model.predict(features_for_model)
                except Exception:
                    logging.exception("Failed to generate predictions for %s", target)
                    continue
                target_values.loc[mask] = preds
            player_predictions[f"pred_{target}"] = target_values.values

        for column in [
            "pred_passing_yards",
            "pred_rushing_yards",
            "pred_receiving_yards",
            "pred_receptions",
            "pred_rushing_tds",
            "pred_receiving_tds",
            "pred_passing_tds",
        ]:
            if column not in player_predictions.columns:
                player_predictions[column] = np.nan

        value_columns = [
            "pred_passing_yards",
            "pred_rushing_yards",
            "pred_receiving_yards",
            "pred_receptions",
            "pred_rushing_tds",
            "pred_receiving_tds",
            "pred_passing_tds",
        ]
        player_predictions[value_columns] = player_predictions[value_columns].fillna(0.0)

        qb_mask = player_predictions["position"] == "QB"
        rb_mask = player_predictions["position"] == "RB"
        wr_mask = player_predictions["position"] == "WR"
        te_mask = player_predictions["position"] == "TE"

        # Quarterbacks: retain passing and rushing yardage, suppress receiving metrics
        player_predictions.loc[qb_mask, [
            "pred_receiving_yards",
            "pred_receptions",
            "pred_receiving_tds",
        ]] = 0.0

        # Rushing output is meaningful for quarterbacks and running backs.
        rushing_allowed_mask = qb_mask | rb_mask
        player_predictions.loc[~rushing_allowed_mask, "pred_rushing_yards"] = 0.0
        player_predictions.loc[~rushing_allowed_mask, "pred_rushing_tds"] = 0.0

        # Pass catchers (RB/WR/TE) retain receiving metrics; others zeroed out
        receivers_mask = rb_mask | wr_mask | te_mask
        player_predictions.loc[~receivers_mask, [
            "pred_receiving_yards",
            "pred_receptions",
            "pred_receiving_tds",
        ]] = 0.0

        # Non-quarterbacks should not have passing output
        player_predictions.loc[~qb_mask, [
            "pred_passing_yards",
            "pred_passing_tds",
        ]] = 0.0

        player_predictions["pred_touchdowns"] = (
            player_predictions["pred_rushing_tds"].fillna(0)
            + player_predictions["pred_receiving_tds"].fillna(0)
            + player_predictions["pred_passing_tds"].fillna(0)
        )

    # Reporting output
    def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]], aligns=None) -> List[str]:
        if aligns is None:
            aligns = ["left"] * len(headers)

        widths = [len(str(header)) for header in headers]
        for row in rows:
            for idx, cell in enumerate(row):
                widths[idx] = max(widths[idx], len(str(cell)))

        def _fmt(cell: str, width: int, align: str) -> str:
            text = str(cell)
            if align == "right":
                return text.rjust(width)
            if align == "center":
                return text.center(width)
            return text.ljust(width)

        sep = "+".join([""] + ["-" * (w + 2) for w in widths] + [""])
        header_line = "| " + " | ".join(
            _fmt(header, width, align)
            for header, width, align in zip(headers, widths, aligns)
        ) + " |"

        table_lines = [sep, header_line, sep]
        for row in rows:
            formatted = "| " + " | ".join(
                _fmt(cell, width, align)
                for cell, width, align in zip(row, widths, aligns)
            ) + " |"
            table_lines.append(formatted)
        table_lines.append(sep)
        return table_lines

    lines: List[str] = []

    if not scoreboard.empty:
        scoreboard_rows: List[List[str]] = []
        for row in scoreboard.itertuples(index=False):
            away_low = row.away_score_lower if not pd.isna(row.away_score_lower) else row.away_score
            away_high = row.away_score_upper if not pd.isna(row.away_score_upper) else row.away_score
            home_low = row.home_score_lower if not pd.isna(row.home_score_lower) else row.home_score
            home_high = row.home_score_upper if not pd.isna(row.home_score_upper) else row.home_score

            scoreboard_rows.append(
                [
                    str(row.date),
                    row.away_team_abbr,
                    row.home_team_abbr,
                    f"{row.away_score:.2f} ({away_low:.2f}-{away_high:.2f})",
                    f"{row.home_score:.2f} ({home_low:.2f}-{home_high:.2f})",
                ]
            )

        lines.extend(
            _format_table(
                ["Date", "Away", "Home", "Away Score ±RMSE", "Home Score ±RMSE"],
                scoreboard_rows,
                aligns=["left", "left", "left", "right", "right"],
            )
        )

    if winner_unc:
        lines.append("")
        lines.append("Game winner calibration:")
        if winner_unc.get("log_loss") is not None:
            lines.append(f"  Log loss: {winner_unc['log_loss']:.3f}")
        if winner_unc.get("brier") is not None:
            lines.append(f"  Brier score: {winner_unc['brier']:.3f}")
        if winner_unc.get("accuracy") is not None:
            lines.append(f"  Validation accuracy: {winner_unc['accuracy']:.3f}")

    position_order = {"QB": 0, "RB": 1, "HB": 1, "FB": 1, "WR": 2, "TE": 3}

    if not player_predictions.empty:
        for game in scoreboard.itertuples(index=False):
            lines.append("")
            lines.append(f"{game.away_team_abbr} at {game.home_team_abbr}")
            lines.append("".ljust(len(lines[-1]), "-"))

            game_players = player_predictions[player_predictions["game_id"] == game.game_id]
            for team in [game.away_team_abbr, game.home_team_abbr]:
                team_players = game_players[game_players["team"] == team].copy()
                if team_players.empty:
                    continue

                team_players["_pos_order"] = team_players["position"].map(position_order).fillna(9)
                team_players = team_players.sort_values(
                    ["_pos_order", "pred_touchdowns", "pred_receptions"], ascending=[True, False, False]
                )

                rows: List[List[str]] = []
                for player in team_players.itertuples(index=False):
                    name = player.player_name or "Unknown Player"

                    rushing_tds = getattr(player, "pred_rushing_tds", 0.0)
                    if pd.isna(rushing_tds):
                        rushing_tds = 0.0

                    display_touchdowns = player.pred_touchdowns
                    if pd.isna(display_touchdowns):
                        display_touchdowns = 0.0

                    if player.position == "QB":
                        display_touchdowns = rushing_tds

                    rows.append(
                        [
                            name,
                            player.position,
                            f"{player.pred_passing_yards:.2f}",
                            f"{player.pred_rushing_yards:.2f}",
                            f"{player.pred_receiving_yards:.2f}",
                            f"{player.pred_receptions:.2f}",
                            f"{display_touchdowns:.2f}",
                            f"{player.pred_passing_tds:.2f}",
                        ]
                    )

                if rows:
                    lines.append("")
                    lines.append(f"{team} Starters")
                    lines.extend(
                        _format_table(
                            [
                                "Player",
                                "Pos",
                                "Pass Yds",
                                "Rush Yds",
                                "Rec Yds",
                                "Receptions",
                                "Rush TDs",
                                "Pass TDs",
                            ],
                            rows,
                            aligns=[
                                "left",
                                "center",
                                "right",
                                "right",
                                "right",
                                "right",
                                "right",
                                "right",
                            ],
                        )
                    )

    report_text = "\n".join(lines)
    print(report_text)

    if save_json and output_path is not None:
        games_payload = scoreboard.copy()
        games_payload["start_time"] = games_payload["start_time"].astype(str)
        if "local_start_time" in games_payload.columns:
            games_payload["local_start_time"] = games_payload["local_start_time"].astype(str)

        players_payload = player_predictions.drop(
            columns=[col for col in player_predictions.columns if col.startswith("_")],
            errors="ignore",
        )
        if "start_time" in players_payload.columns:
            players_payload["start_time"] = players_payload["start_time"].astype(str)
        if "local_start_time" in players_payload.columns:
            players_payload["local_start_time"] = players_payload["local_start_time"].astype(str)

        output_payload = {
            "games": games_payload.to_dict(orient="records"),
            "players": players_payload.to_dict(orient="records"),
        }
        output_path.write_text(json.dumps(output_payload, indent=2))
        logging.info("Saved prediction summary to %s", output_path)

    return {"games": scoreboard, "players": player_predictions}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NFL betting analytics pipeline")
    parser.add_argument("--config", type=str, help="Optional path to JSON config file")
    parser.add_argument("--predict", action="store_true", help="Generate predictions for upcoming games")
    parser.add_argument("--output", type=Path, default=Path("predictions.json"), help="Where to save predictions")
    parser.add_argument(
        "--respect-lineups",
        action="store_true",
        default=None,
        help="Filter modeling and props to starters from MSF lineup.json when available",
    )
    return parser.parse_args()


def load_config(path: Optional[str]) -> NFLConfig:
    config = NFLConfig()
    if not path:
        return config

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    for field in dataclasses.fields(config):
        if field.name in payload:
            setattr(config, field.name, payload[field.name])
    return config


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if getattr(args, "respect_lineups", None) is not None:
        config.respect_lineups = bool(args.respect_lineups)
    setup_logging(config.log_level)

    logging.info("Connecting to PostgreSQL at %s", config.pg_url)
    engine = create_engine(config.pg_url, future=True)
    db = NFLDatabase(engine)

    msf_client = MySportsFeedsClient(NFL_API_USER, NFL_API_PASS)
    odds_client = OddsApiClient(ODDS_API_KEY)
    supplemental_loader = SupplementalDataLoader(config)

    ingestor = NFLIngestor(db, msf_client, odds_client, supplemental_loader)
    ingestor.ingest(config.seasons)

    trainer = ModelTrainer(engine, db)
    try:
        models = trainer.train()
    except RuntimeError as exc:
        logging.error("Unable to train models: %s", exc)
        logging.error(
            "Model training requires historical games and player statistics. "
            "Ensure ingestion succeeded (check API credentials, plan access, and season settings) before rerunning."
        )
        if args.predict:
            logging.error("Prediction generation skipped because models were not trained.")
        return

    if not models:
        logging.warning(
            "No models were trained. Verify that sufficient labeled data exists in the database before requesting predictions."
        )
        if args.predict:
            logging.error("Prediction generation skipped because no models were available.")
        return

    predict_upcoming_games(
        models,
        engine,
        trainer.model_uncertainty,
        output_path=args.output if args.predict else None,
        save_json=args.predict,
        ingestor=ingestor,
        trainer=trainer,
        config=config,
    )


if __name__ == "__main__":
    main()
