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
import asyncio
import dataclasses
import datetime as dt
import io
import json
import logging
import math
import os
import re
import ssl
import sys
import time
import unicodedata
import uuid
from collections import defaultdict
from types import SimpleNamespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
from zoneinfo import ZoneInfo
from html import unescape

import aiohttp
import numpy as np
import pandas as pd
import requests
import zipfile
from aiohttp import client_exceptions
import certifi
from requests import HTTPError
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from requests.exceptions import (
    ConnectionError as RequestsConnectionError,
    JSONDecodeError as RequestsJSONDecodeError,
    ReadTimeout,
    SSLError,
)
from http import HTTPStatus
from sklearn import set_config
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
    r2_score,
    roc_auc_score,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neighbors import NearestNeighbors
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
    and_,
    create_engine,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from urllib.parse import urlencode, urljoin
import urllib3
from urllib3.exceptions import InsecureRequestWarning
from urllib3.util import Retry

try:  # Optional dependency used for HTML parsing when available
    from bs4 import BeautifulSoup, Tag  # type: ignore
except Exception:  # pragma: no cover - fallback when bs4 is absent
    BeautifulSoup = None  # type: ignore
    Tag = None  # type: ignore

_BEAUTIFULSOUP_WARNING_EMITTED = False


SCRIPT_ROOT = Path(__file__).resolve().parent


def _default_data_file(name: str) -> Optional[str]:
    candidate = SCRIPT_ROOT / "data" / name
    if candidate.exists():
        return str(candidate)
    return None


DEFAULT_CLOSING_ODDS_PATH = _default_data_file("closing_odds_history.csv")
DEFAULT_TRAVEL_CONTEXT_PATH = _default_data_file("team_travel_context.csv")


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


# ---------------------------------------------------------------------------
# Closing odds archive downloaders
# ---------------------------------------------------------------------------


MONEYLINE_PATTERN = re.compile(r"([-+]?\d+)")

SEASON_COLUMN_CANDIDATES = ["season", "yr", "year"]
WEEK_COLUMN_CANDIDATES = ["week", "wk", "weeknum", "week_no", "game_week"]
HOME_TEAM_COLUMN_CANDIDATES = [
    "home",
    "home_team",
    "home_team_name",
    "home_team_abbr",
    "home_club",
    "home_name",
    "team_home",
    "home side",
    "homeclub",
]
AWAY_TEAM_COLUMN_CANDIDATES = [
    "away",
    "away_team",
    "away_team_name",
    "away_team_abbr",
    "visitor",
    "visitor_team",
    "road",
    "opp",
    "opponent",
    "team_away",
    "vis",
]
TEAM_COLUMN_CANDIDATES = ["team", "team_name", "club", "squad"]
OPPONENT_COLUMN_CANDIDATES = ["opponent", "opp", "opp_name", "opponent_name"]
SITE_COLUMN_CANDIDATES = [
    "site",
    "homeaway",
    "home_away",
    "ha",
    "venue_type",
    "location",
]
DATE_COLUMN_CANDIDATES = ["date", "game_date", "start_date", "schedule_date"]
TIME_COLUMN_CANDIDATES = ["time", "start_time", "game_time", "kickoff", "kickoff_time"]
HOME_MONEYLINE_COLUMN_CANDIDATES = [
    "home_ml",
    "home_moneyline",
    "home_money_line",
    "moneyline_home",
    "ml_home",
    "home_close_ml",
    "home_close_moneyline",
    "home_closing_ml",
    "home_closing_moneyline",
    "home_close",
]
AWAY_MONEYLINE_COLUMN_CANDIDATES = [
    "away_ml",
    "away_moneyline",
    "away_money_line",
    "moneyline_away",
    "ml_away",
    "road_ml",
    "road_moneyline",
    "visitor_ml",
    "visitor_moneyline",
    "vis_ml",
    "away_close_ml",
    "away_close_moneyline",
    "away_closing_moneyline",
]
TEAM_MONEYLINE_COLUMN_CANDIDATES = [
    "ml",
    "moneyline",
    "money_line",
    "close_ml",
    "team_ml",
    "team_moneyline",
]
BOOKMAKER_COLUMN_CANDIDATES = ["book", "sportsbook", "bookmaker", "source", "sportsbook_name"]


def _canonicalize_column_name(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(label or "").strip().lower())


def _match_column(columns: Dict[str, str], candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    for candidate in candidates:
        for key, original in columns.items():
            if candidate in key:
                return original
    return None


def _parse_moneyline_series(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    text = series.astype(str).str.strip().str.replace(",", "", regex=False)
    text = text.replace({"nan": np.nan, "": np.nan, "None": np.nan})
    extracted = text.str.extract(MONEYLINE_PATTERN)
    numeric = pd.to_numeric(extracted[0], errors="coerce")
    return numeric


def _combine_date_time(
    date_series: Optional[pd.Series], time_series: Optional[pd.Series]
) -> pd.Series:
    if date_series is None:
        return pd.Series(dtype="datetime64[ns, UTC]")
    date_values = pd.to_datetime(date_series, errors="coerce")
    if time_series is not None:
        combined = (
            date_series.astype(str).str.strip()
            + " "
            + time_series.astype(str).str.strip()
        )
        combined_dt = pd.to_datetime(combined, errors="coerce")
        if hasattr(combined_dt, "combine_first"):
            date_values = combined_dt.combine_first(date_values)
    try:
        if hasattr(date_values.dt, "tz") and date_values.dt.tz is None:
            date_values = date_values.dt.tz_localize("UTC")
    except (TypeError, AttributeError, ValueError):
        try:
            date_values = date_values.dt.tz_localize(
                "UTC", nonexistent="NaT", ambiguous="NaT"
            )
        except Exception:
            date_values = date_values.dt.tz_localize(None)
    return date_values


def _reshape_team_based_closing_rows(
    frame: pd.DataFrame,
    provider_name: str,
    *,
    season_hint: Optional[str],
    season_col: Optional[str],
    week_col: Optional[str],
    date_col: Optional[str],
    time_col: Optional[str],
    site_col: Optional[str],
    team_col: Optional[str],
    opp_col: Optional[str],
    ml_col: Optional[str],
    book_col: Optional[str],
) -> pd.DataFrame:
    if not all([team_col, opp_col, ml_col]):
        logging.warning(
            "%s dataset is missing explicit team/opponent columns and cannot be reshaped",
            provider_name,
        )
        return pd.DataFrame()

    working = frame.copy()
    working["_team"] = working[team_col].apply(normalize_team_abbr)
    working["_opp"] = working[opp_col].apply(normalize_team_abbr)
    working["_ml"] = _parse_moneyline_series(working[ml_col])

    if site_col:
        site_values = (
            working[site_col].astype(str).str.strip().str.upper()
        )
    else:
        site_values = pd.Series("", index=working.index)
    working["_site"] = site_values
    working["_is_home"] = working["_site"].str.startswith("H")
    working["_is_away"] = working["_site"].str.startswith("A")

    if season_col:
        working["_season"] = working[season_col]
    else:
        working["_season"] = season_hint

    if week_col:
        working["_week"] = pd.to_numeric(working[week_col], errors="coerce")
    else:
        working["_week"] = np.nan

    kickoff_series = _combine_date_time(
        working[date_col] if date_col else None,
        working[time_col] if time_col else None,
    )
    working["_kickoff"] = kickoff_series

    working["_key"] = working.apply(
        lambda row: _compose_game_key(
            row.get("_season"),
            row.get("_week"),
            row.get("_kickoff"),
            row.get("_team"),
            row.get("_opp"),
        ),
        axis=1,
    )

    records: List[Dict[str, Any]] = []
    for _key, group in working.groupby("_key"):
        group = group.dropna(subset=["_team", "_opp"])
        if group.empty:
            continue

        home_row = group[group["_is_home"]].head(1)
        away_row = group[group["_is_away"]].head(1)

        chosen_home: Optional[pd.Series] = home_row.iloc[0] if not home_row.empty else None
        chosen_away: Optional[pd.Series] = None

        if chosen_home is not None:
            counterpart = group[group["_team"] == chosen_home["_opp"]]
            if not counterpart.empty:
                chosen_away = counterpart.iloc[0]

        if chosen_home is None and not away_row.empty:
            candidate_away = away_row.iloc[0]
            counterpart = group[group["_team"] == candidate_away["_opp"]]
            if not counterpart.empty:
                chosen_home = counterpart.iloc[0]
                chosen_away = candidate_away

        if chosen_home is None or chosen_away is None:
            pair_found = False
            for _, row in group.iterrows():
                counterpart = group[group["_team"] == row["_opp"]]
                if counterpart.empty:
                    continue
                other = counterpart.iloc[0]
                candidate_home, candidate_away = row, other
                if candidate_home["_is_away"] and not candidate_away["_is_away"]:
                    candidate_home, candidate_away = candidate_away, candidate_home
                elif candidate_away["_is_home"] and not candidate_home["_is_home"]:
                    candidate_home, candidate_away = candidate_away, candidate_home
                chosen_home = candidate_home
                chosen_away = candidate_away
                pair_found = True
                break
            if not pair_found:
                logging.warning(
                    "%s dataset could not resolve complementary home/away rows for game %s",
                    provider_name,
                    _key,
                )
                continue

        if chosen_home is None or chosen_away is None:
            continue

        home_team = chosen_home.get("_team")
        away_team = chosen_away.get("_team")
        home_ml = chosen_home.get("_ml")
        away_ml = chosen_away.get("_ml")

        if pd.isna(home_team) or pd.isna(away_team):
            continue
        if pd.isna(home_ml) and pd.isna(away_ml):
            continue

        if pd.isna(home_ml) and not pd.isna(away_ml):
            logging.debug(
                "%s dataset is missing a home moneyline for matchup %s; skipping",
                provider_name,
                _key,
            )
            continue
        if pd.isna(away_ml) and not pd.isna(home_ml):
            logging.debug(
                "%s dataset is missing an away moneyline for matchup %s; skipping",
                provider_name,
                _key,
            )
            continue

        season_value = chosen_home.get("_season")
        week_value = _safe_week_value(chosen_home.get("_week"))
        kickoff_value = chosen_home.get("_kickoff")
        if (kickoff_value is None or pd.isna(kickoff_value)) and chosen_away is not None:
            alt_kickoff = chosen_away.get("_kickoff")
            if alt_kickoff is not None and not pd.isna(alt_kickoff):
                kickoff_value = alt_kickoff

        def _resolve_season(candidate: Any) -> str:
            if candidate is not None and not (
                isinstance(candidate, float) and math.isnan(candidate)
            ):
                text = str(candidate).strip()
                if text and text.lower() not in {"nan", "none", "null"}:
                    return text
            inferred = _infer_regular_season_label_from_timestamp(kickoff_value)
            if inferred:
                return inferred
            if season_hint:
                return str(season_hint)
            return ""

        season_label = _resolve_season(season_value)

        bookmaker_value: Optional[str] = None
        if book_col:
            home_book = chosen_home.get(book_col)
            away_book = chosen_away.get(book_col)
            if isinstance(home_book, str) and home_book.strip():
                bookmaker_value = home_book.strip()
            elif isinstance(away_book, str) and away_book.strip():
                bookmaker_value = away_book.strip()

        records.append(
            {
                "season": season_label,
                "week": week_value,
                "home_team": home_team,
                "away_team": away_team,
                "home_closing_moneyline": home_ml,
                "away_closing_moneyline": away_ml,
                "closing_line_time": kickoff_value,
                "closing_bookmaker": bookmaker_value or provider_name,
            }
        )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame.from_records(records)


def _safe_week_value(value: Any) -> Optional[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _compose_game_key(
    season_value: Any,
    week_value: Any,
    kickoff: Optional[pd.Timestamp],
    home_team: Optional[str],
    away_team: Optional[str],
) -> Tuple[str, Optional[int], str, str]:
    season_str = str(season_value) if season_value not in (None, "") else ""
    week_int = _safe_week_value(week_value)
    if isinstance(kickoff, pd.Timestamp) and not pd.isna(kickoff):
        kickoff_token = kickoff.tz_convert("UTC").strftime("%Y-%m-%d") if kickoff.tzinfo else kickoff.strftime("%Y-%m-%d")
    else:
        kickoff_token = ""
    teams_sorted = sorted(filter(None, [home_team, away_team]))
    team_token = "|".join(teams_sorted)
    return season_str, week_int, kickoff_token, team_token


def _normalize_historical_closing_frame(
    frame: pd.DataFrame,
    provider_name: str,
    season_hint: Optional[str] = None,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    columns = {
        _canonicalize_column_name(col): col for col in frame.columns
    }

    season_col = _match_column(columns, SEASON_COLUMN_CANDIDATES)
    week_col = _match_column(columns, WEEK_COLUMN_CANDIDATES)
    home_col = _match_column(columns, HOME_TEAM_COLUMN_CANDIDATES)
    away_col = _match_column(columns, AWAY_TEAM_COLUMN_CANDIDATES)
    book_col = _match_column(columns, BOOKMAKER_COLUMN_CANDIDATES)
    date_col = _match_column(columns, DATE_COLUMN_CANDIDATES)
    time_col = _match_column(columns, TIME_COLUMN_CANDIDATES)

    team_col = _match_column(columns, TEAM_COLUMN_CANDIDATES)
    opp_col = _match_column(columns, OPPONENT_COLUMN_CANDIDATES)
    site_col = _match_column(columns, SITE_COLUMN_CANDIDATES)
    ml_col = _match_column(columns, TEAM_MONEYLINE_COLUMN_CANDIDATES)

    home_ml_col = _match_column(columns, HOME_MONEYLINE_COLUMN_CANDIDATES)
    away_ml_col = _match_column(columns, AWAY_MONEYLINE_COLUMN_CANDIDATES)

    if team_col and opp_col and ml_col:
        needs_reshape = False
        if not home_col or not away_col:
            needs_reshape = True
        elif not home_ml_col or not away_ml_col:
            needs_reshape = True

        if needs_reshape:
            reshaped = _reshape_team_based_closing_rows(
                frame,
                provider_name,
                season_hint=season_hint,
                season_col=season_col,
                week_col=week_col,
                date_col=date_col,
                time_col=time_col,
                site_col=site_col,
                team_col=team_col,
                opp_col=opp_col,
                ml_col=ml_col,
                book_col=book_col,
            )
            if not reshaped.empty:
                return reshaped

    if not home_ml_col or not away_ml_col:
        logging.warning(
            "%s dataset is missing moneyline columns and cannot be imported",
            provider_name,
        )
        return pd.DataFrame()

    if season_col and season_col in frame.columns:
        season_series: Union[pd.Series, Any] = frame[season_col]
        if isinstance(season_series, pd.Series) and season_hint is not None:
            season_series = season_series.fillna(season_hint)
    else:
        season_series = pd.Series(season_hint, index=frame.index)
    if not isinstance(season_series, pd.Series):
        season_series = pd.Series(
            [season_series] * len(frame), index=frame.index, dtype=object
        )
    else:
        season_series = season_series.astype(object)

    week_series = (
        pd.to_numeric(frame[week_col], errors="coerce") if week_col else np.nan
    )
    kickoff_series = _combine_date_time(
        frame[date_col] if date_col else None,
        frame[time_col] if time_col else None,
    )

    inferred_season = kickoff_series.apply(
        _infer_regular_season_label_from_timestamp
    )
    if season_hint is not None:
        inferred_season = inferred_season.fillna(season_hint)

    def _normalize_season_value(raw_val: Any, inferred_val: Optional[str]) -> str:
        candidate: Optional[str]
        if raw_val is None or (isinstance(raw_val, float) and math.isnan(raw_val)):
            candidate = None
        else:
            text = str(raw_val).strip()
            candidate = (
                text
                if text and text.lower() not in {"nan", "none", "null"}
                else None
            )
        if not candidate:
            candidate = inferred_val
        if not candidate and season_hint:
            candidate = str(season_hint)
        return str(candidate or "")

    result = pd.DataFrame(index=frame.index)
    result["season"] = [
        _normalize_season_value(raw_val, inferred_val)
        for raw_val, inferred_val in zip(season_series, inferred_season)
    ]
    result["week"] = week_series
    result["home_team"] = frame[home_col].apply(normalize_team_abbr)
    result["away_team"] = frame[away_col].apply(normalize_team_abbr)
    result["home_closing_moneyline"] = _parse_moneyline_series(frame[home_ml_col])
    result["away_closing_moneyline"] = _parse_moneyline_series(frame[away_ml_col])
    if book_col:
        result["closing_bookmaker"] = frame[book_col].astype(str).fillna("")
    else:
        result["closing_bookmaker"] = provider_name
    result["closing_line_time"] = kickoff_series

    return result



def _infer_regular_season_label_from_timestamp(value: Any) -> Optional[str]:
    """Infer an NFL regular-season label (e.g. ``2025-regular``) from a date."""

    if value is None:
        return None

    timestamp = pd.to_datetime(value, errors="coerce")
    if isinstance(timestamp, pd.DatetimeIndex):
        if len(timestamp) == 0:
            return None
        timestamp = timestamp[0]
    if pd.isna(timestamp):
        return None

    try:
        year = int(timestamp.year)
        month = int(timestamp.month)
    except Exception:
        return None

    if month < 8:
        year -= 1

    return f"{year}-regular"


def _standardize_closing_odds_frame(
    frame: pd.DataFrame,
    provider_name: str,
    season_hint: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize arbitrary historical closing odds layouts into a standard frame."""

    if frame is None or frame.empty:
        return pd.DataFrame()

    working = frame.copy()
    columns = {_canonicalize_column_name(col): col for col in working.columns}

    season_col = _match_column(columns, SEASON_COLUMN_CANDIDATES)
    if not season_col:
        date_col = _match_column(columns, DATE_COLUMN_CANDIDATES)
        if date_col:
            inferred = pd.to_datetime(working[date_col], errors="coerce")
            working["season"] = inferred.apply(_infer_regular_season_label_from_timestamp)
            season_col = "season"
        elif season_hint:
            working["season"] = season_hint
            season_col = "season"

    if season_col and season_col != "season":
        working = working.rename(columns={season_col: "season"})

    normalized = _normalize_historical_closing_frame(
        working, provider_name, season_hint
    )
    if normalized.empty:
        return pd.DataFrame()

    normalized = normalized.copy()
    normalized["season"] = normalized["season"].astype(str)
    if "week" in normalized.columns:
        normalized["week"] = pd.to_numeric(normalized["week"], errors="coerce")
    if "closing_line_time" in normalized.columns:
        normalized["closing_line_time"] = pd.to_datetime(
            normalized["closing_line_time"], errors="coerce", utc=True
        )
    else:
        normalized["closing_line_time"] = pd.NaT

    if "closing_bookmaker" in normalized.columns:
        normalized["closing_bookmaker"] = normalized["closing_bookmaker"].fillna(
            provider_name
        )
    else:
        normalized["closing_bookmaker"] = provider_name

    return normalized


def _oddsportal_slug_candidates(season_label: str) -> List[str]:
    """Return likely OddsPortal result slugs for a given season label."""
    default_slug = "nfl/results/"
    label = str(season_label or "").strip()
    candidates: List[str] = []
    if default_slug not in candidates:
        candidates.append(default_slug)

    digit_tokens = [tok for tok in re.findall(r"\d{4}", label)]
    for token in digit_tokens:
        try:
            year = int(token)
        except ValueError:
            continue
        next_year = year + 1
        for slug in (
            f"nfl-{year}-{next_year}/results/",
            f"nfl-{year}-{str(next_year)[-2:]}/results/",
            f"nfl-{year}/results/",
        ):
            if slug not in candidates:
                candidates.append(slug)

    normalized = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    if normalized and normalized not in {"nfl", "usa", "results"}:
        slug = normalized
        if not slug.endswith("/results") and not slug.endswith("/results/"):
            slug = f"{slug}/results/"
        elif slug.endswith("/results"):
            slug = f"{slug}/"
        if slug not in candidates:
            candidates.append(slug)

    return candidates


def _parse_decimal_odds(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        decimal = float(match.group(0))
    except ValueError:
        return None
    if not math.isfinite(decimal) or decimal <= 1.0:
        return None
    return decimal


def _decimal_to_american(decimal: Optional[float]) -> Optional[int]:
    if decimal is None or not math.isfinite(decimal) or decimal <= 1.0:
        return None
    if decimal >= 2.0:
        return int(round((decimal - 1.0) * 100.0))
    try:
        return int(round(-100.0 / (decimal - 1.0)))
    except ZeroDivisionError:
        return None


def _parse_moneyline_text(value: Optional[str]) -> Optional[int]:
    """Parse an odds string expressed as either American or decimal prices."""

    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    normalized = text.replace("\xa0", " ")
    american_match = re.search(r"[+-]\d{2,4}", normalized)
    if american_match:
        try:
            return int(american_match.group(0))
        except ValueError:
            return None

    bare_match = re.search(r"\b\d{3,4}\b", normalized)
    if bare_match:
        try:
            candidate = int(bare_match.group(0))
        except ValueError:
            candidate = None
        else:
            if candidate is not None and candidate >= 100:
                return candidate

    decimal = _parse_decimal_odds(normalized)
    if decimal is not None:
        return _decimal_to_american(decimal)

    return None


def _parse_oddsportal_datetime(
    date_text: Optional[str],
    time_text: Optional[str],
    *,
    timezone: str = "UTC",
) -> Optional[pd.Timestamp]:
    pieces = []
    if date_text:
        pieces.append(str(date_text).strip())
    if time_text:
        pieces.append(str(time_text).strip())
    if not pieces:
        return None
    combined = " ".join(pieces)
    parsed = pd.to_datetime(combined, errors="coerce")
    if parsed is pd.NaT or parsed is None:
        return None
    if isinstance(parsed, pd.DatetimeIndex):
        parsed = parsed.to_series().iloc[0]
    if getattr(parsed, "tzinfo", None) is None:
        try:
            parsed = parsed.tz_localize(timezone)
        except Exception:
            try:
                parsed = parsed.tz_localize("UTC", nonexistent="NaT", ambiguous="NaT")
            except Exception:
                return None
    else:
        try:
            parsed = parsed.tz_convert(timezone)
        except Exception:
            pass
    return parsed


class LocalClosingOddsFetcher:
    """Load pre-downloaded closing odds from a local CSV file."""

    def __init__(self, csv_path: Optional[str]) -> None:
        self.csv_path = Path(csv_path).expanduser() if csv_path else None

    def fetch(self, seasons: Sequence[str]) -> pd.DataFrame:
        if not self.csv_path:
            logging.warning(
                "Local closing odds path not configured; set NFL_CLOSING_ODDS_PATH to a CSV file"
            )
            return pd.DataFrame()

        try:
            if not self.csv_path.exists():
                logging.warning("Local closing odds file %s not found", self.csv_path)
                return pd.DataFrame()
        except OSError:
            logging.exception(
                "Unable to access local closing odds file at %s", self.csv_path
            )
            return pd.DataFrame()

        try:
            raw = pd.read_csv(self.csv_path)
        except Exception:
            logging.exception(
                "Failed to read closing odds CSV from %s", self.csv_path
            )
            return pd.DataFrame()

        normalized = _standardize_closing_odds_frame(raw, "Local CSV")
        if normalized.empty:
            logging.warning(
                "Local closing odds file %s did not produce any usable rows", self.csv_path
            )
            return normalized

        normalized = normalized.copy()
        normalized["season"] = normalized["season"].astype(str)
        if seasons:
            wanted = {str(season) for season in seasons}
            normalized = normalized[normalized["season"].isin(wanted)]
        if "week" in normalized.columns:
            normalized["week"] = pd.to_numeric(normalized["week"], errors="coerce")

        return normalized.reset_index(drop=True)


class OddsPortalFetcher:
    """Scrape historical closing odds from OddsPortal results pages."""

    def __init__(
        self,
        session: requests.Session,
        *,
        base_url: str = "https://www.oddsportal.com/american-football/usa/",
        results_path: str = "nfl/results/",
        season_path_template: str = "nfl-{season}/results/",
        timeout: int = 45,
        user_agents: Optional[Sequence[str]] = None,
    ) -> None:
        self.session = session
        self.base_url = (base_url or "https://www.oddsportal.com/american-football/usa/").strip()
        if not self.base_url.endswith("/"):
            self.base_url += "/"
        self.results_path = results_path.strip("/") + "/" if results_path else "nfl/results/"
        self.season_path_template = season_path_template
        self.timeout = timeout

        candidates: List[str] = []
        for ua in list(user_agents or []):
            cleaned = (ua or "").strip()
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)
        for default in (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:118.0) Gecko/20100101 Firefox/118.0",
        ):
            if default not in candidates:
                candidates.append(default)
        if not candidates:
            candidates.append("Mozilla/5.0")
        self.user_agents = candidates
        self._insecure_notice_logged = False
        self._insecure_success_logged = False
        self._ssl_failure_logged = False
        self._insecure_adapter_installed = False

        self._debug_dump_enabled = False
        self._debug_dir: Optional[Path] = None
        self._debug_dumped_sources: Set[str] = set()
        self._debug_capture_logged: Set[str] = set()
        self._no_rows_warned: Set[str] = set()

        debug_flag = os.environ.get("NFL_ODDSPORTAL_DEBUG_HTML", "")
        if str(debug_flag).strip().lower() in {"1", "true", "yes", "on", "debug"}:
            self._debug_dump_enabled = True
            debug_dir = os.environ.get("NFL_ODDSPORTAL_DEBUG_DIR")
            self._debug_dir = (
                Path(debug_dir).expanduser()
                if debug_dir
                else SCRIPT_ROOT / "reports" / "oddsportal_debug"
            )
            try:
                self._debug_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                logging.exception(
                    "Unable to create OddsPortal debug directory at %s", self._debug_dir
                )
                self._debug_dump_enabled = False
            else:
                logging.warning(
                    "NFL_ODDSPORTAL_DEBUG_HTML enabled; raw OddsPortal pages will be written to %s",
                    self._debug_dir,
                )

        if BeautifulSoup is None:
            raise RuntimeError(
                "The beautifulsoup4 package is required to scrape OddsPortal closing odds. "
                "Install it with 'pip install beautifulsoup4' inside your environment or disable "
                "OddsPortal scraping in your configuration if you do not need closing odds."
            )

    def fetch(self, seasons: Sequence[str]) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for season in seasons:
            season_label = str(season)
            season_frame = self._fetch_season(season_label)
            if not _is_effectively_empty_df(season_frame):
                frames.append(season_frame)
            elif self._ssl_failure_logged:
                # If TLS negotiation failed even after insecure retries,
                # avoid hammering every slug for the same season.
                break
        return safe_concat(frames, ignore_index=True)

    def _fetch_season(self, season_label: str) -> pd.DataFrame:
        slugs = self._season_slugs(season_label)
        frames: List[pd.DataFrame] = []
        for slug in slugs:
            data = self._scrape_slug(slug, season_label)
            if not _is_effectively_empty_df(data):
                frames.append(data)
                if slug != slugs[0]:
                    logging.debug(
                        "OddsPortal used fallback slug %s for season %s", slug, season_label
                    )
                break
        return safe_concat(frames, ignore_index=True)

    def _season_slugs(self, season_label: str) -> List[str]:
        slugs = []
        base_slug = self.results_path
        if base_slug and base_slug not in slugs:
            slugs.append(base_slug)
        for candidate in _oddsportal_slug_candidates(season_label):
            if candidate not in slugs:
                slugs.append(candidate)
        template = (self.season_path_template or "nfl-{season}/results/").strip()
        if "{season}" in template:
            sanitized = re.sub(r"[^0-9a-zA-Z]+", "-", str(season_label).strip()).strip("-")
            if sanitized:
                slug = template.format(season=sanitized)
                slug = slug.strip("/") + "/"
                if slug not in slugs:
                    slugs.append(slug)
        return slugs

    def _scrape_slug(self, slug: str, season_label: str) -> pd.DataFrame:
        url = urljoin(self.base_url, slug)
        html = self._request(url)
        if not html:
            return pd.DataFrame()

        frames: List[pd.DataFrame] = []
        parsed = self._parse_results_page(html, season_label)
        if not _is_effectively_empty_df(parsed):
            frames.append(parsed)
        else:
            self._debug_capture_failure(slug, html, source_url=url)

        for page_url in self._discover_additional_pages(url, html):
            page_html = self._request(page_url)
            if not page_html:
                continue
            chunk = self._parse_results_page(page_html, season_label)
            if not _is_effectively_empty_df(chunk):
                frames.append(chunk)
            else:
                self._debug_capture_failure(slug, page_html, source_url=page_url)

        result = safe_concat(frames, ignore_index=True)
        if _is_effectively_empty_df(result):
            self._debug_warn_no_rows(slug, season_label, url)
        return result

    def _request(self, url: str) -> Optional[str]:
        attempt_insecure = False

        base_headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Referer": self.base_url,
            "Upgrade-Insecure-Requests": "1",
        }

        def _build_headers(user_agent: str, *, json_variant: bool) -> Dict[str, str]:
            headers = dict(base_headers)
            headers["User-Agent"] = user_agent
            if json_variant:
                headers.update(
                    {
                        "Accept": "application/json, text/plain, */*",
                        "X-Requested-With": "XMLHttpRequest",
                        "x-nextjs-data": "1",
                    }
                )
            else:
                headers.setdefault("Sec-Fetch-Site", "same-origin")
                headers.setdefault("Sec-Fetch-Mode", "navigate")
                headers.setdefault("Sec-Fetch-Dest", "document")
            return headers

        for ua in self.user_agents:
            try:
                response = self.session.get(
                    url, headers=_build_headers(ua, json_variant=False), timeout=self.timeout
                )
            except SSLError as exc:
                logging.error("OddsPortal SSL error for %s: %s", url, exc)
                attempt_insecure = True
                break
            except Exception:
                logging.exception("OddsPortal request error for %s", url)
                response = None

            if response is not None:
                result = self._process_oddsportal_response(response, url, insecure=False)
                if result is not None:
                    return result

            try:
                response = self.session.get(
                    url,
                    headers=_build_headers(ua, json_variant=True),
                    timeout=self.timeout,
                )
            except SSLError as exc:
                logging.error("OddsPortal SSL error for %s (JSON variant): %s", url, exc)
                attempt_insecure = True
                break
            except Exception:
                logging.exception("OddsPortal JSON request error for %s", url)
                continue

            result = self._process_oddsportal_response(response, url, insecure=False)
            if result is not None:
                return result

        fallback_attempted = False

        if attempt_insecure and self.session.verify is not False:
            fallback_attempted = True
            if not self._insecure_notice_logged:
                logging.warning(
                    "Falling back to insecure HTTPS for OddsPortal after certificate verification "
                    "failed. Provide NFL_ODDS_SSL_CERT or set ODDS_ALLOW_INSECURE_SSL=true to "
                    "avoid this automatic downgrade."
                )
                self._insecure_notice_logged = True
            # persist the fallback for subsequent requests
            self._install_insecure_adapter()
            self.session.verify = False
            for ua in self.user_agents:
                try:
                    response = self.session.get(
                        url,
                        headers=_build_headers(ua, json_variant=False),
                        timeout=self.timeout,
                        verify=False,
                    )
                except SSLError as exc:
                    logging.error(
                        "OddsPortal SSL error persisted for %s even with verification disabled: %s",
                        url,
                        exc,
                    )
                    response = None
                except Exception:
                    logging.exception("OddsPortal request error for %s", url)
                    response = None

                if response is not None:
                    result = self._process_oddsportal_response(response, url, insecure=True)
                    if result is not None:
                        return result

                try:
                    response = self.session.get(
                        url,
                        headers=_build_headers(ua, json_variant=True),
                        timeout=self.timeout,
                        verify=False,
                    )
                except SSLError as exc:
                    logging.error(
                        "OddsPortal SSL error persisted for %s (JSON variant) even with verification disabled: %s",
                        url,
                        exc,
                    )
                    continue
                except Exception:
                    logging.exception("OddsPortal JSON request error for %s", url)
                    continue

                result = self._process_oddsportal_response(response, url, insecure=True)
                if result is not None:
                    return result

        if attempt_insecure and not self._ssl_failure_logged and (
            not fallback_attempted or self.session.verify is False
        ):
            logging.warning(
                "Certificate verification failed when contacting OddsPortal. "
                "If you are behind a corporate proxy, set NFL_ODDS_SSL_CERT to a trusted CA "
                "bundle or temporarily enable ODDS_ALLOW_INSECURE_SSL=true."
            )
            self._ssl_failure_logged = True

        return None

    def _install_insecure_adapter(self) -> None:
        """Mount a requests adapter that disables SSL verification when required."""

        if self._insecure_adapter_installed:
            return

        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=None,
        )

        try:
            adapter = HTTPAdapter(max_retries=retry, ssl_context=context)
        except TypeError:
            logging.debug(
                "requests.HTTPAdapter does not support the ssl_context argument; "
                "falling back to a legacy-compatible adapter"
            )
            adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        urllib3.disable_warnings(InsecureRequestWarning)
        self._insecure_adapter_installed = True

    def _process_oddsportal_response(
        self, response: requests.Response, url: str, *, insecure: bool
    ) -> Optional[str]:
        if response.status_code == HTTPStatus.OK:
            text = response.text or ""
            if text.strip():
                if insecure and not self._insecure_success_logged:
                    logging.warning(
                        "OddsPortal request for %s succeeded only after disabling certificate "
                        "verification. Supply NFL_ODDS_SSL_CERT or set ODDS_ALLOW_INSECURE_SSL=true "
                        "to acknowledge this behaviour.",
                        url,
                    )
                    self._insecure_success_logged = True
                return text

        if response.status_code in {HTTPStatus.FORBIDDEN, HTTPStatus.NOT_FOUND}:
            logging.warning(
                "OddsPortal request for %s returned status %s", url, response.status_code
            )
            return None

        logging.debug(
            "OddsPortal request for %s returned status %s", url, response.status_code
        )
        return None

    def _extract_json_score(self, entry: Dict[str, Any]) -> Optional[float]:
        values: List[float] = []

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    lower = str(key).lower()
                    if any(token in lower for token in ("score", "points", "result")):
                        if isinstance(value, (int, float)):
                            values.append(float(value))
                        elif isinstance(value, str):
                            match = re.search(r"-?\d+(?:\.\d+)?", value)
                            if match:
                                try:
                                    values.append(float(match.group(0)))
                                except ValueError:
                                    pass
                        elif isinstance(value, (dict, list)):
                            visit(value)
                    elif isinstance(value, (dict, list)):
                        visit(value)
            elif isinstance(node, list):
                for item in node:
                    visit(item)

        visit(entry)

        return values[0] if values else None

    def _extract_json_kickoff(self, ancestors: Sequence[Dict[str, Any]]) -> Optional[pd.Timestamp]:
        for container in reversed(ancestors):
            if not isinstance(container, dict):
                continue
            for key, value in container.items():
                lower = str(key).lower()
                if any(
                    token in lower
                    for token in (
                        "start",
                        "kickoff",
                        "commence",
                        "date",
                        "time",
                        "begin",
                        "eventtime",
                        "timestamp",
                    )
                ):
                    timestamp = self._convert_json_datetime(value)
                    if timestamp is not None:
                        return timestamp

    def _parse_results_page(self, html: str, season_label: str) -> pd.DataFrame:
        global _BEAUTIFULSOUP_WARNING_EMITTED
        if BeautifulSoup is None:
            if not _BEAUTIFULSOUP_WARNING_EMITTED:
                logging.warning(
                    "BeautifulSoup is required to parse OddsPortal pages; install beautifulsoup4"
                )
                _BEAUTIFULSOUP_WARNING_EMITTED = True
            return pd.DataFrame()

        soup = BeautifulSoup(html, "html.parser")
        table = soup.find(class_=re.compile(r"\btable-main\b"))
        if table is None:
            table = soup.find(id=re.compile("tournamentTable", re.IGNORECASE))
        if table is None:
            modern_rows = self._parse_modern_results(soup, season_label)
            if not modern_rows.empty:
                return modern_rows
            state_rows = self._parse_embedded_state(html, season_label, soup=soup)
            if not state_rows.empty:
                return state_rows
            try:
                frames = pd.read_html(io.StringIO(html))
            except Exception:
                frames = []
            if not frames:
                return pd.DataFrame()
            # Fallback: attempt to normalise first readable table
            for frame in frames:
                if frame.empty:
                    continue
                normalized = self._normalise_table(frame, season_label)
                if not normalized.empty:
                    return normalized
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        for node in table.find_all(class_=re.compile("event__match")):
            header = node.find_previous(class_=re.compile("event__header"))
            current_date = None
            if header is not None:
                for ancestor in header.parents:
                    if ancestor is table:
                        current_date = header.get_text(" ", strip=True)
                        break

            home_node = node.find(class_=re.compile("event__participant--home"))
            away_node = node.find(class_=re.compile("event__participant--away"))
            if home_node is None or away_node is None:
                continue
            home_team = normalize_team_abbr(home_node.get_text(" ", strip=True))
            away_team = normalize_team_abbr(away_node.get_text(" ", strip=True))

            time_node = node.find(class_=re.compile("event__time"))
            scores_node = node.find(class_=re.compile("event__scores"))
            odds_nodes = node.find_all(class_=re.compile("(odd|odds)"))

            kickoff = _parse_oddsportal_datetime(
                current_date,
                time_node.get_text(strip=True) if time_node else None,
            )

            home_score = away_score = np.nan
            if scores_node:
                score_match = re.findall(r"\d+", scores_node.get_text(" ", strip=True))
                if len(score_match) >= 2:
                    home_score = float(score_match[0])
                    away_score = float(score_match[1])

            decimals: List[float] = []
            for odds in odds_nodes:
                value = _parse_decimal_odds(odds.get_text(" ", strip=True))
                if value is not None:
                    decimals.append(value)
            home_decimal = decimals[0] if decimals else None
            away_decimal = decimals[-1] if len(decimals) > 1 else None

            rows.append(
                {
                    "season": season_label,
                    "week": np.nan,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_closing_moneyline": _decimal_to_american(home_decimal),
                    "away_closing_moneyline": _decimal_to_american(away_decimal),
                    "closing_bookmaker": "OddsPortal",
                    "closing_line_time": kickoff,
                    "kickoff_utc": kickoff,
                    "kickoff_date": kickoff.strftime("%Y-%m-%d") if kickoff else "",
                    "kickoff_weekday": kickoff.strftime("%a") if kickoff else "",
                    "home_score": home_score,
                    "away_score": away_score,
                }
            )

        if rows:
            return pd.DataFrame(rows)

        modern_rows = self._parse_modern_results(soup, season_label)
        if not modern_rows.empty:
            return modern_rows
        state_rows = self._parse_embedded_state(html, season_label, soup=soup)
        if not state_rows.empty:
            return state_rows

        try:
            frames = pd.read_html(io.StringIO(html))
        except Exception:
            return pd.DataFrame()
        for frame in frames:
            if frame.empty:
                continue
            normalized = self._normalise_table(frame, season_label)
            if not normalized.empty:
                return normalized

        return pd.DataFrame()

    def _debug_capture_failure(self, slug: str, html: str, *, source_url: str) -> None:
        if not html:
            return

        key = f"{slug}|{source_url}"
        if key in self._debug_capture_logged:
            return
        self._debug_capture_logged.add(key)

        html_bytes = len(html)
        legacy_nodes = modern_rows = participant_nodes = next_data_scripts = 0
        json_like = False

        if BeautifulSoup is not None:
            try:
                soup = BeautifulSoup(html, "html.parser")
            except Exception as exc:
                logging.debug(
                    "OddsPortal diagnostics unable to parse HTML for %s (%s): %s",
                    source_url,
                    slug,
                    exc,
                )
            else:
                legacy_nodes = len(soup.find_all(class_=re.compile("event__match")))
                modern_rows = len(self._find_by_testid(soup, "game-row"))
                participant_nodes = len(self._find_by_testid(soup, "event-participants"))
                next_data_scripts = len(soup.find_all(id="__NEXT_DATA__"))
                if not legacy_nodes and not modern_rows:
                    snippet = html.lstrip()
                    json_like = snippet.startswith("{") or snippet.startswith("[")
                else:
                    json_like = False
        else:
            snippet = html.lstrip()
            json_like = snippet.startswith("{") or snippet.startswith("[")

        logging.info(
            "OddsPortal parse diagnostics for %s -> %s: html_bytes=%d legacy_nodes=%d modern_rows=%d participant_nodes=%d next_data_scripts=%d json_like=%s",
            slug,
            source_url,
            html_bytes,
            legacy_nodes,
            modern_rows,
            participant_nodes,
            next_data_scripts,
            json_like,
        )

        self._debug_dump_html(slug, source_url, html)

    def _debug_dump_html(self, slug: str, source_url: str, html: str) -> None:
        if not self._debug_dump_enabled or not html or self._debug_dir is None:
            return

        key = f"{slug}|{source_url}"
        if key in self._debug_dumped_sources:
            return
        self._debug_dumped_sources.add(key)

        timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        safe_slug = re.sub(r"[^0-9A-Za-z]+", "-", slug.strip("/")) or "root"
        filename = f"{timestamp}_{safe_slug}.html"
        path = self._debug_dir / filename

        try:
            path.write_text(html, encoding="utf-8")
        except Exception:
            logging.exception("Failed to write OddsPortal debug HTML to %s", path)
            self._debug_dump_enabled = False
            return

        logging.warning(
            "Saved OddsPortal HTML snapshot to %s for slug %s (%s). Share this file if parsing remains empty.",
            path,
            slug,
            source_url,
        )

    def _debug_warn_no_rows(self, slug: str, season_label: str, source_url: str) -> None:
        key = f"{season_label}|{slug}"
        if key in self._no_rows_warned:
            return

        self._no_rows_warned.add(key)
        logging.warning(
            "OddsPortal parser did not find closing odds for slug %s (season %s, url=%s). Set NFL_ODDSPORTAL_DEBUG_HTML=1 and rerun to capture the raw HTML for troubleshooting.",
            slug,
            season_label,
            source_url,
        )

    def _tag_testid_values(self, tag: "Tag") -> Sequence[str]:
        values: List[str] = []
        if not isinstance(tag, Tag):
            return values

        for attr, raw_value in tag.attrs.items():
            try:
                attr_name = str(attr).lower()
            except Exception:
                continue

            if not attr_name.startswith("data-test"):
                continue

            def _append(value: Any) -> None:
                if value is None:
                    return
                if isinstance(value, (list, tuple, set)):
                    for item in value:
                        _append(item)
                else:
                    try:
                        values.append(str(value))
                    except Exception:
                        pass

            _append(raw_value)

        return values

    def _tag_has_testid(self, tag: "Tag", targets: Sequence[str]) -> bool:
        if not isinstance(tag, Tag):
            return False

        normalized_targets = [t.strip().lower() for t in targets if t]
        if not normalized_targets:
            return False

        for value in self._tag_testid_values(tag):
            token = value.strip().lower()
            if not token:
                continue
            for target in normalized_targets:
                if not target:
                    continue
                if token == target:
                    return True
                if token.startswith(target):
                    return True
                if target in token:
                    return True
        return False

    def _find_by_testid(self, root: "BeautifulSoup", *targets: str) -> List["Tag"]:
        if BeautifulSoup is None or Tag is None:
            return []

        normalized_targets = [t for t in targets if t]

        def matcher(node: Any) -> bool:
            return self._tag_has_testid(node, normalized_targets)

        return list(root.find_all(matcher)) if normalized_targets else []

    def _parse_modern_results(self, soup: "BeautifulSoup", season_label: str) -> pd.DataFrame:
        if Tag is None:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []

        for node in self._find_by_testid(soup, "game-row"):
            if not isinstance(node, Tag):
                continue

            parent = node.parent
            skip = False
            while isinstance(parent, Tag):
                if self._tag_has_testid(parent, ["game-row"]):
                    skip = True
                    break
                parent = parent.parent
            if skip:
                continue

            container = node if node.name != "a" else node.parent
            if not isinstance(container, Tag):
                container = node

            participants = None
            for candidate in self._find_by_testid(container, "event-participants"):
                participants = candidate
                break
            if not isinstance(participants, Tag):
                continue

            score_pattern = re.compile(r"^-?\d+$")

            def _extract_team(tag: "Tag") -> Tuple[str, Optional[float]]:
                raw_name = (tag.get("title") or "").strip()
                if not raw_name:
                    name_node = tag.find(class_=re.compile("participant-name"))
                    if isinstance(name_node, Tag):
                        raw_name = name_node.get_text(" ", strip=True)
                    elif tag.name == "p" and re.search("participant-name", " ".join(tag.get("class", []))):
                        raw_name = tag.get_text(" ", strip=True)
                    else:
                        raw_name = tag.get_text(" ", strip=True)

                team = normalize_team_abbr(raw_name)
                if not team:
                    return "", None

                score_value: Optional[float] = None
                for candidate in tag.find_all(True):
                    if not isinstance(candidate, Tag):
                        continue
                    text = candidate.get_text("", strip=True)
                    if score_pattern.fullmatch(text):
                        try:
                            score_value = float(text)
                        except ValueError:
                            score_value = None
                        break
                if score_value is None:
                    for token in tag.stripped_strings:
                        text = token.strip()
                        if score_pattern.fullmatch(text):
                            try:
                                score_value = float(text)
                            except ValueError:
                                score_value = None
                            break

                if score_value is None:
                    sibling = tag
                    for _ in range(3):
                        sibling = getattr(sibling, "next_sibling", None)
                        while isinstance(sibling, str) and not sibling.strip():
                            sibling = getattr(sibling, "next_sibling", None)
                        if not isinstance(sibling, Tag):
                            continue
                        text = sibling.get_text("", strip=True)
                        if score_pattern.fullmatch(text):
                            try:
                                score_value = float(text)
                            except ValueError:
                                score_value = None
                            break
                        for token in sibling.stripped_strings:
                            t = token.strip()
                            if score_pattern.fullmatch(t):
                                try:
                                    score_value = float(t)
                                except ValueError:
                                    score_value = None
                                break
                        if score_value is not None:
                            break

                return team, score_value

            teams: List[Tuple[str, Optional[float]]] = []
            for link in participants.find_all("a"):
                if not isinstance(link, Tag):
                    continue
                team, score = _extract_team(link)
                if not team:
                    continue
                teams.append((team, score))
                if len(teams) >= 2:
                    break

            if len(teams) < 2:
                for name_node in participants.find_all(class_=re.compile("participant-name")):
                    if not isinstance(name_node, Tag):
                        continue
                    team, score = _extract_team(name_node)
                    if not team:
                        continue
                    teams.append((team, score))
                    if len(teams) >= 2:
                        break

            if len(teams) < 2:
                continue

            away_team, away_score = teams[0]
            home_team, home_score = teams[1]

            if not away_team or not home_team:
                continue

            time_node: Optional[Tag] = None
            for candidate in self._find_by_testid(container, "time-item"):
                time_node = candidate
                break
            time_text = (
                time_node.get_text(" ", strip=True)
                if isinstance(time_node, Tag)
                else None
            )

            date_text = self._find_associated_date(container)
            kickoff = _parse_oddsportal_datetime(date_text, time_text)

            odds_values: List[int] = []
            for odd_container in container.find_all(True):
                if not isinstance(odd_container, Tag):
                    continue
                if not (
                    self._tag_has_testid(odd_container, ["odd-container"])
                    or any(
                        str(attr or "").lower().startswith("data-odd-container")
                        for attr in odd_container.attrs.keys()
                    )
                ):
                    continue
                price = _parse_moneyline_text(odd_container.get_text(" ", strip=True))
                if price is None:
                    continue
                odds_values.append(price)
                if len(odds_values) >= 2:
                    break

            away_moneyline = odds_values[0] if odds_values else None
            home_moneyline = odds_values[1] if len(odds_values) > 1 else None

            rows.append(
                {
                    "season": season_label,
                    "week": np.nan,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_closing_moneyline": home_moneyline,
                    "away_closing_moneyline": away_moneyline,
                    "closing_bookmaker": "OddsPortal",
                    "closing_line_time": kickoff,
                    "kickoff_utc": kickoff,
                    "kickoff_date": kickoff.strftime("%Y-%m-%d") if kickoff else "",
                    "kickoff_weekday": kickoff.strftime("%a") if kickoff else "",
                    "home_score": home_score if home_score is not None else np.nan,
                    "away_score": away_score if away_score is not None else np.nan,
                }
            )

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    def _parse_embedded_state(
        self,
        html: str,
        season_label: str,
        *,
        soup: Optional["BeautifulSoup"] = None,
    ) -> pd.DataFrame:
        if BeautifulSoup is None:
            return pd.DataFrame()

        if soup is None:
            soup = BeautifulSoup(html, "html.parser")

        payloads = self._extract_json_payloads(html, soup)
        if not payloads:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        seen: Set[Tuple[Any, ...]] = set()

        for payload in payloads:
            data = self._safe_json_loads(payload)
            if data is None:
                continue

            name_cache: Dict[int, Optional[str]] = {}
            for ancestors, _, participant_list in self._iter_json_candidate_lists(
                data, name_cache
            ):
                normalized = self._normalise_json_participants(participant_list, name_cache)
                if len(normalized) < 2:
                    continue

                valid_entries = [entry for entry in normalized if entry.get("team")]
                if len({entry["team"] for entry in valid_entries}) < 2:
                    continue

                has_odds = any(entry.get("moneyline") is not None for entry in valid_entries)
                if not has_odds:
                    continue

                home_entry = next((e for e in valid_entries if e.get("is_home") is True), None)
                away_entry = next((e for e in valid_entries if e.get("is_home") is False), None)

                if home_entry is None or away_entry is None:
                    ordered = sorted(
                        valid_entries,
                        key=lambda item: (
                            0 if item.get("is_home") is False else 1 if item.get("is_home") is True else 2,
                            item.get("index", 0),
                        ),
                    )
                    if len(ordered) >= 2:
                        away_entry, home_entry = ordered[0], ordered[1]
                    else:
                        continue

                kickoff = self._extract_json_kickoff(ancestors)

                key = (
                    home_entry.get("team"),
                    away_entry.get("team"),
                    kickoff.isoformat() if isinstance(kickoff, pd.Timestamp) else kickoff,
                    home_entry.get("moneyline"),
                    away_entry.get("moneyline"),
                )
                if key in seen:
                    continue
                seen.add(key)

                rows.append(
                    {
                        "season": season_label,
                        "week": np.nan,
                        "home_team": home_entry.get("team"),
                        "away_team": away_entry.get("team"),
                        "home_closing_moneyline": home_entry.get("moneyline"),
                        "away_closing_moneyline": away_entry.get("moneyline"),
                        "closing_bookmaker": "OddsPortal",
                        "closing_line_time": kickoff,
                        "kickoff_utc": kickoff,
                        "kickoff_date": kickoff.strftime("%Y-%m-%d") if isinstance(kickoff, pd.Timestamp) else "",
                        "kickoff_weekday": kickoff.strftime("%a") if isinstance(kickoff, pd.Timestamp) else "",
                        "home_score": home_entry.get("score", np.nan),
                        "away_score": away_entry.get("score", np.nan),
                    }
                )

        if not rows:
            return pd.DataFrame()

        frame = pd.DataFrame(rows)
        frame["kickoff_utc"] = pd.to_datetime(frame["kickoff_utc"], utc=True, errors="coerce")
        frame["closing_line_time"] = frame["kickoff_utc"]
        frame["kickoff_date"] = frame["kickoff_utc"].dt.strftime("%Y-%m-%d").fillna("")
        frame["kickoff_weekday"] = frame["kickoff_utc"].dt.strftime("%a").fillna("")
        frame["home_score"] = pd.to_numeric(frame["home_score"], errors="coerce")
        frame["away_score"] = pd.to_numeric(frame["away_score"], errors="coerce")
        frame["home_score"] = frame["home_score"].where(frame["home_score"].notna(), np.nan)
        frame["away_score"] = frame["away_score"].where(frame["away_score"].notna(), np.nan)

        return frame.reset_index(drop=True)

    def _extract_json_payloads(
        self, html: str, soup: Optional["BeautifulSoup"]
    ) -> List[str]:
        payloads: List[str] = []

        stripped = html.lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            payloads.append(stripped)

        if soup is not None:
            for script in soup.find_all("script"):
                text = script.string or script.get_text()
                if not text:
                    continue
                if script.get("id") == "__NEXT_DATA__":
                    payloads.append(text)
                    continue
                snippet = text.strip()
                if any(token in snippet for token in ("__NEXT_DATA__", "__NUXT__", "eventGroup")):
                    payloads.append(snippet)

        patterns = [
            r"<script[^>]+id=\"__NEXT_DATA__\"[^>]*>(.*?)</script>",
            r"window\.__NEXT_DATA__\s*=\s*(\{.*?\})\s*;",
            r"window\.__NUXT__\s*=\s*(\{.*?\})\s*;",
            r"window\.__INITIAL_STATE__\s*=\s*(\{.*?\})\s*;",
            r"data-state=(?:\"|\')(\{.*?\})(?:\"|\')",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, html, flags=re.DOTALL):
                payloads.append(match.group(1))

        unique: List[str] = []
        seen: Set[str] = set()
        for payload in payloads:
            if not payload:
                continue
            cleaned = unescape(payload).strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            unique.append(cleaned)

        return unique

    def _safe_json_loads(self, payload: str) -> Optional[Any]:
        text = (payload or "").strip()
        if not text:
            return None

        text = unescape(text)
        text = text.strip()

        for prefix in ("window.__NEXT_DATA__", "window.__NUXT__", "window.__INITIAL_STATE__"):
            if text.startswith(prefix):
                parts = text.split("=", 1)
                text = parts[1] if len(parts) == 2 else ""
                break

        text = text.strip()
        if text.startswith("export default"):
            text = text[len("export default") :].strip()

        if text.startswith("const ") or text.startswith("var "):
            parts = text.split("=", 1)
            text = parts[1] if len(parts) == 2 else ""

        text = text.strip()
        if text.endswith(";"):
            text = text[:-1]

        text = text.strip()
        if not text:
            return None

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            sanitized = re.sub(r"//.*?$", "", text, flags=re.MULTILINE).strip()
            sanitized = sanitized.strip(";")
            if not sanitized:
                return None
            try:
                return json.loads(sanitized)
            except json.JSONDecodeError:
                return None

    def _iter_json_candidate_lists(
        self, data: Any, name_cache: Dict[int, Optional[str]]
    ) -> Iterable[Tuple[List[Dict[str, Any]], Optional[str], List[Dict[str, Any]]]]:
        if not isinstance(data, (dict, list)):
            return []

        visited_lists: Set[int] = set()
        stack: List[Tuple[List[Dict[str, Any]], Any]] = [([], data)]

        while stack:
            ancestors, node = stack.pop()

            if isinstance(node, dict):
                for key, value in node.items():
                    if isinstance(value, list):
                        if id(value) not in visited_lists:
                            visited_lists.add(id(value))
                            if self._looks_like_participant_list(value, name_cache):
                                yield ancestors + [node], key, value  # type: ignore[arg-type]
                            stack.append((ancestors + [node], value))
                    elif isinstance(value, (dict, list)):
                        stack.append((ancestors + [node], value))
            elif isinstance(node, list):
                for item in node:
                    stack.append((ancestors, item))

    def _looks_like_participant_list(
        self, value: Sequence[Any], name_cache: Dict[int, Optional[str]]
    ) -> bool:
        if not isinstance(value, Sequence) or len(value) < 2:
            return False

        dict_count = sum(1 for item in value if isinstance(item, dict))
        if dict_count < 2:
            return False

        identified = 0
        for item in value:
            if not isinstance(item, dict):
                continue
            name = self._extract_json_team_name(item, name_cache)
            if name:
                identified += 1
            if identified >= 2:
                return True

        return False

    def _normalise_json_participants(
        self, entries: Sequence[Any], name_cache: Dict[int, Optional[str]]
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []

        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue

            name = self._extract_json_team_name(entry, name_cache)
            if not name:
                continue

            team = normalize_team_abbr(name)
            if not team:
                continue

            normalized.append(
                {
                    "team": team,
                    "is_home": self._infer_json_home_indicator(entry),
                    "moneyline": self._extract_json_odds(entry),
                    "score": self._extract_json_score(entry),
                    "index": index,
                }
            )

        return normalized

    def _extract_json_team_name(
        self, node: Any, cache: Dict[int, Optional[str]]
    ) -> Optional[str]:
        queue: List[Any] = [node]
        seen: Set[int] = set()

        while queue:
            current = queue.pop(0)
            node_id = id(current)
            if node_id in seen:
                continue
            seen.add(node_id)

            if node_id in cache:
                cached = cache[node_id]
                if cached:
                    return cached
                continue

            if isinstance(current, dict):
                for key, value in current.items():
                    lower = str(key).lower()
                    if any(
                        token in lower
                        for token in (
                            "name",
                            "team",
                            "participant",
                            "competitor",
                            "club",
                            "title",
                            "slug",
                            "abbr",
                            "abbreviation",
                        )
                    ):
                        if isinstance(value, str):
                            cleaned = value.strip()
                            if cleaned and any(ch.isalpha() for ch in cleaned):
                                cache[node_id] = cleaned
                                return cleaned
                        elif isinstance(value, (dict, list)):
                            queue.append(value)
                            continue
                    if isinstance(value, (dict, list)):
                        queue.append(value)

                cache[node_id] = None

            elif isinstance(current, list):
                queue.extend(current)

        return None

    def _infer_json_home_indicator(self, entry: Dict[str, Any]) -> Optional[bool]:
        for key, value in entry.items():
            lower = str(key).lower()
            if lower in {"ishome", "home"}:
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    if value == 1:
                        return True
                    if value == 0 or value == -1:
                        return False
                if isinstance(value, str):
                    token = value.strip().lower()
                    if token.startswith("home") or token in {"h", "host"}:
                        return True
                    if token.startswith("away") or token in {"a", "visitor", "road", "guest"}:
                        return False

        for key in (
            "homeAway",
            "home_away",
            "homeAwayTeam",
            "qualifier",
            "alignment",
            "side",
            "designation",
            "teamType",
            "teamRole",
            "venueType",
        ):
            if key not in entry:
                continue
            value = entry[key]
            if isinstance(value, str):
                token = value.strip().lower()
                if "home" in token and "away" not in token:
                    return True
                if any(tag in token for tag in ("away", "road", "guest", "visitor")):
                    return False
            elif isinstance(value, dict):
                nested = value.get("value") or value.get("label") or value.get("name")
                if isinstance(nested, str):
                    nested_token = nested.strip().lower()
                    if "home" in nested_token and "away" not in nested_token:
                        return True
                    if any(tag in nested_token for tag in ("away", "road", "guest", "visitor")):
                        return False

        indicator = entry.get("isHome")
        if isinstance(indicator, bool):
            return indicator

        return None

    def _extract_json_odds(self, entry: Dict[str, Any]) -> Optional[int]:
        candidates: List[Tuple[bool, Optional[int]]] = []

        def visit(node: Any, *, keyed: bool = False) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    lower = str(key).lower()
                    is_odds_key = any(
                        token in lower
                        for token in (
                            "american",
                            "moneyline",
                            "money_line",
                            "price",
                            "odds",
                            "us",
                        )
                    )
                    if isinstance(value, (str, int, float)) and is_odds_key:
                        price = _parse_moneyline_text(str(value))
                        candidates.append((True, price))
                    elif isinstance(value, (dict, list)):
                        visit(value, keyed=is_odds_key or keyed)
            elif isinstance(node, list):
                for item in node:
                    visit(item, keyed=keyed)
            elif keyed and isinstance(node, (str, int, float)):
                price = _parse_moneyline_text(str(node))
                candidates.append((False, price))

        visit(entry)

        for keyed, price in candidates:
            if price is not None:
                return price

        return None

    def _extract_json_score(self, entry: Dict[str, Any]) -> Optional[float]:
        values: List[float] = []

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    lower = str(key).lower()
                    if any(token in lower for token in ("score", "points", "result")):
                        if isinstance(value, (int, float)):
                            values.append(float(value))
                        elif isinstance(value, str):
                            match = re.search(r"-?\d+(?:\.\d+)?", value)
                            if match:
                                try:
                                    values.append(float(match.group(0)))
                                except ValueError:
                                    pass
                        elif isinstance(value, (dict, list)):
                            visit(value)
                    elif isinstance(value, (dict, list)):
                        visit(value)
            elif isinstance(node, list):
                for item in node:
                    visit(item)

        visit(entry)

        return values[0] if values else None

    def _extract_json_kickoff(self, ancestors: Sequence[Dict[str, Any]]) -> Optional[pd.Timestamp]:
        for container in reversed(ancestors):
            if not isinstance(container, dict):
                continue
            for key, value in container.items():
                lower = str(key).lower()
                if any(
                    token in lower
                    for token in (
                        "start",
                        "kickoff",
                        "commence",
                        "date",
                        "time",
                        "begin",
                        "eventtime",
                        "timestamp",
                    )
                ):
                    timestamp = self._convert_json_datetime(value)
                    if timestamp is not None:
                        return timestamp

        return None

    def _convert_json_datetime(self, value: Any) -> Optional[pd.Timestamp]:
        if value is None:
            return None

        if isinstance(value, pd.Timestamp):
            return value.tz_convert("UTC") if value.tzinfo else value.tz_localize("UTC")

        if isinstance(value, (int, float)):
            if value > 1e12:
                timestamp = pd.to_datetime(value, unit="ms", utc=True, errors="coerce")
            else:
                timestamp = pd.to_datetime(value, unit="s", utc=True, errors="coerce")
            if pd.notna(timestamp):
                return timestamp
            return None

        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                timestamp = pd.to_datetime(text, utc=True, errors="coerce")
            except Exception:
                timestamp = pd.NaT
            if pd.notna(timestamp):
                return timestamp
            digits = re.sub(r"\D", "", text)
            if digits:
                try:
                    numeric = int(digits)
                except ValueError:
                    numeric = None
                if numeric is not None:
                    if len(digits) >= 13:
                        timestamp = pd.to_datetime(numeric, unit="ms", utc=True, errors="coerce")
                    else:
                        timestamp = pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
                    if pd.notna(timestamp):
                        return timestamp

        if isinstance(value, dict):
            for key in ("iso", "utc", "value", "timestamp"):
                if key in value:
                    candidate = self._convert_json_datetime(value[key])
                    if candidate is not None:
                        return candidate

        return None

    def _find_associated_date(self, node: "Tag") -> Optional[str]:
        if Tag is None:
            return None

        date_pattern = re.compile(r"(date|day|header)", re.IGNORECASE)
        month_pattern = re.compile(
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)", re.IGNORECASE
        )

        current: Optional[Tag] = node
        while current is not None:
            sibling = current.previous_sibling
            while sibling is not None:
                if isinstance(sibling, Tag):
                    match_found = False
                    if sibling.has_attr("data-testid"):
                        attr_val = str(sibling.get("data-testid", ""))
                        if date_pattern.search(attr_val):
                            match_found = True
                    if not match_found:
                        for value in self._tag_testid_values(sibling):
                            if date_pattern.search(value):
                                match_found = True
                                break
                    if match_found:
                        text = sibling.get_text(" ", strip=True)
                        if text:
                            return text
                    if sibling.get("class") and any(
                        date_pattern.search(str(cls)) for cls in sibling.get("class", [])
                    ):
                        text = sibling.get_text(" ", strip=True)
                        if text:
                            return text
                    text = sibling.get_text(" ", strip=True)
                    if text and (
                        month_pattern.search(text)
                        or re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", text)
                    ):
                        return text
                sibling = getattr(sibling, "previous_sibling", None)
            current = current.parent if isinstance(getattr(current, "parent", None), Tag) else None

        for candidate in node.find_all_previous(True, limit=25):
            if not isinstance(candidate, Tag):
                continue
            text = candidate.get_text(" ", strip=True)
            if not text:
                continue
            if month_pattern.search(text) or re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", text):
                return text

        return None

    def _normalise_table(self, frame: pd.DataFrame, season_label: str) -> pd.DataFrame:
        frame = frame.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [
                " ".join(str(part) for part in col if str(part) != "nan").strip()
                for col in frame.columns
            ]
        frame = frame.loc[:, ~frame.columns.astype(str).str.contains(r"^Unnamed", case=False)]
        frame.columns = [str(col).strip() for col in frame.columns]

        lower_cols = {col.lower(): col for col in frame.columns}
        home_col = None
        away_col = None
        for key in ("home", "home team", "home_team"):
            if key in lower_cols:
                home_col = lower_cols[key]
                break
        for key in ("away", "away team", "away_team"):
            if key in lower_cols:
                away_col = lower_cols[key]
                break

        match_col = None
        if not home_col or not away_col:
            for key in ("match", "event", "teams", "matchup", "home - away"):
                if key in lower_cols:
                    match_col = lower_cols[key]
                    break
            if match_col is None:
                for col in frame.columns:
                    sample = (
                        frame[col]
                        .astype(str)
                        .str.contains(r"\b(vs|vs\.|@| - | – | v )\b", case=False, regex=True)
                    )
                    if sample.any():
                        match_col = col
                        break

        odds_columns: List[str] = []
        for col in frame.columns:
            parsed = frame[col].apply(lambda val: _parse_decimal_odds(str(val)))
            if parsed.notna().sum() >= max(1, int(len(frame) * 0.4)):
                odds_columns.append(col)
        if not odds_columns:
            odds_columns = [
                col
                for col in frame.columns
                if frame[col]
                .astype(str)
                .str.contains(r"\d\.\d", regex=True)
                .sum()
                >= max(1, int(len(frame) * 0.4))
            ]

        home_decimal = None
        away_decimal = None
        for key in ("1", "home odds", "home_odds"):
            if key in lower_cols:
                home_decimal = lower_cols[key]
                break
        for key in ("2", "away odds", "away_odds"):
            if key in lower_cols:
                away_decimal = lower_cols[key]
                break

        if home_decimal is None and odds_columns:
            home_decimal = odds_columns[0]
        if away_decimal is None and len(odds_columns) >= 2:
            away_decimal = odds_columns[1]

        def _split_teams(value: Any) -> Tuple[str, str]:
            text = str(value or "").strip()
            separators = [" - ", " – ", " vs ", " vs. ", " v ", " @ "]
            for sep in separators:
                if sep in text:
                    parts = [part.strip() for part in text.split(sep, 1)]
                    if len(parts) == 2:
                        return parts[0], parts[1]
            tokens = re.split(r"\s+vs\.?\s+|\s+@\s+", text, maxsplit=1, flags=re.IGNORECASE)
            if len(tokens) == 2:
                return tokens[0].strip(), tokens[1].strip()
            return "", ""

        if match_col and (not home_col or not away_col):
            extracted = frame[match_col].apply(_split_teams)
            frame["__home"] = extracted.apply(lambda pair: pair[0])
            frame["__away"] = extracted.apply(lambda pair: pair[1])
            home_col = home_col or "__home"
            away_col = away_col or "__away"

        if not home_col or not away_col:
            return pd.DataFrame()

        result = pd.DataFrame()
        result["season"] = season_label
        result["week"] = np.nan
        result["home_team"] = frame[home_col].apply(normalize_team_abbr)
        result["away_team"] = frame[away_col].apply(normalize_team_abbr)

        def _convert_series(col: Optional[str]) -> pd.Series:
            if not col or col not in frame.columns:
                return pd.Series(np.nan, index=frame.index)
            return frame[col].apply(
                lambda val: _decimal_to_american(_parse_decimal_odds(str(val)))
            )

        result["home_closing_moneyline"] = _convert_series(home_decimal)
        result["away_closing_moneyline"] = _convert_series(away_decimal)
        result["closing_bookmaker"] = "OddsPortal"
        result["closing_line_time"] = pd.NaT
        result["kickoff_utc"] = pd.NaT
        result["kickoff_date"] = ""
        result["kickoff_weekday"] = ""
        result["home_score"] = np.nan
        result["away_score"] = np.nan

        score_col = None
        for key in ("score", "result", "ft", "full time"):
            if key in lower_cols:
                score_col = lower_cols[key]
                break
        if score_col and score_col in frame.columns:
            scores = frame[score_col].astype(str).apply(lambda text: re.findall(r"\d+", text))
            result["home_score"] = scores.apply(lambda vals: float(vals[0]) if len(vals) >= 1 else np.nan)
            result["away_score"] = scores.apply(lambda vals: float(vals[1]) if len(vals) >= 2 else np.nan)

        return result.reset_index(drop=True)


def _season_param_from_label(label: str) -> Optional[str]:
    """Normalize a season label for providers expecting year or year-range strings."""

    text = (label or "").strip()
    if not text:
        return None

    years = re.findall(r"\d{4}", text)
    if not years:
        return text

    if len(years) >= 2:
        return f"{years[0]}-{years[1]}"

    year = years[0]
    try:
        next_year = str(int(year) + 1)
    except ValueError:
        return year
    return f"{year}-{next_year}"


class KillerSportsFetcher:
    def __init__(
        self,
        session: requests.Session,
        *,
        base_url: Optional[str] = None,
        timeout: int = 45,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.session = session
        self.base_url = base_url
        self.timeout = timeout
        self.api_key = api_key
        self.auth = HTTPBasicAuth(username, password) if username and password else None

    def fetch(self, seasons: Sequence[str]) -> pd.DataFrame:
        if not self.base_url:
            logging.warning(
                "KillerSports provider configured but no base URL supplied; skipping download"
            )
            return pd.DataFrame()

        frames: List[pd.DataFrame] = []
        for season in seasons:
            frame = self._fetch_season(str(season))
            if not _is_effectively_empty_df(frame):
                frames.append(frame)
        return safe_concat(frames, ignore_index=True)

    def _fetch_season(self, season: str) -> pd.DataFrame:
        season_param = _season_param_from_label(season)
        if season_param and season_param != season:
            logging.debug(
                "KillerSports season label %s sanitized to %s", season, season_param
            )
        params = {"season": season_param or season, "format": "csv"}
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request_kwargs = dict(
            params=params,
            headers=headers,
            timeout=self.timeout,
            auth=self.auth,
        )
        try:
            response = self.session.get(self.base_url, **request_kwargs)
            response.raise_for_status()
        except SSLError as exc:
            logging.error("KillerSports SSL error for season %s: %s", season, exc)
            if self.session.verify is not False:
                logging.warning(
                    "Falling back to insecure HTTPS for KillerSports after certificate "
                    "verification failed. Provide NFL_ODDS_SSL_CERT or set "
                    "ODDS_ALLOW_INSECURE_SSL=true to avoid this downgrade."
                )
                try:
                    response = self.session.get(self.base_url, verify=False, **request_kwargs)
                    response.raise_for_status()
                except SSLError as insecure_exc:
                    logging.error(
                        "KillerSports SSL error persisted for season %s even with verification "
                        "disabled: %s",
                        season,
                        insecure_exc,
                    )
                    logging.warning(
                        "Certificate verification failed when contacting KillerSports. Set "
                        "NFL_ODDS_SSL_CERT to a trusted CA bundle or use "
                        "ODDS_ALLOW_INSECURE_SSL=true if you accept the risk."
                    )
                    return pd.DataFrame()
                except Exception:
                    logging.exception(
                        "KillerSports request encountered an error for season %s", season
                    )
                    return pd.DataFrame()
                else:
                    self.session.verify = False
                    logging.warning(
                        "KillerSports request for season %s succeeded only after disabling "
                        "certificate verification.",
                        season,
                    )
            else:
                logging.warning(
                    "Certificate verification failed when contacting KillerSports. Set "
                    "NFL_ODDS_SSL_CERT to a trusted CA bundle or use "
                    "ODDS_ALLOW_INSECURE_SSL=true if you accept the risk."
                )
                return pd.DataFrame()
        except HTTPError as exc:
            logging.warning("KillerSports request failed for season %s: %s", season, exc)
            return pd.DataFrame()
        except Exception:
            logging.exception(
                "KillerSports request encountered an error for season %s", season
            )
            return pd.DataFrame()

        try:
            payload = pd.read_csv(io.BytesIO(response.content))
        except Exception:
            try:
                payload = pd.read_html(io.BytesIO(response.content))[0]
            except Exception:
                logging.warning(
                    "KillerSports response for season %s was not a readable CSV/HTML table",
                    season,
                )
                return pd.DataFrame()

        normalized = _normalize_historical_closing_frame(payload, "KillerSports", season)
        if normalized.empty:
            logging.warning(
                "KillerSports data for season %s could not be normalized", season
            )
        return normalized


class ClosingOddsArchiveSyncer:
    def __init__(self, config: "NFLConfig", db: "NFLDatabase") -> None:
        self.config = config
        self.db = db
        self.session = requests.Session()
        if config.odds_ssl_cert_path:
            self.session.verify = config.odds_ssl_cert_path
            logging.info(
                "Using custom SSL certificate bundle for odds downloads: %s",
                config.odds_ssl_cert_path,
            )
        elif config.odds_allow_insecure_ssl:
            self.session.verify = False
            logging.warning(
                "ODDS_ALLOW_INSECURE_SSL enabled; HTTPS verification is disabled"
                " for odds archive downloads. Use only if a corporate proxy"
                " prevents certificate validation."
            )
        else:
            self.session.verify = certifi.where()

    def sync(self) -> None:
        provider = (self.config.closing_odds_provider or "").strip().lower()
        if not provider:
            provider = "oddsportal"
        if provider in {"none", "off", "disable", "disabled"}:
            logging.info(
                "Closing odds archive sync disabled via NFL_CLOSING_ODDS_PROVIDER=%s",
                provider,
            )
            return

        try:
            seasons = [str(season) for season in self.config.seasons]

            local_provider = False

            if provider in {"local", "csv", "file", "history", "offline"}:
                fetcher = LocalClosingOddsFetcher(self.config.closing_odds_history_path)
                provider_name = "Local CSV"
                local_provider = True
            elif provider in {"oddsportal", "odds-portal", "op"}:
                fetcher = OddsPortalFetcher(
                    self.session,
                    base_url=self.config.oddsportal_base_url,
                    results_path=self.config.oddsportal_results_path,
                    season_path_template=self.config.oddsportal_season_template,
                    timeout=self.config.closing_odds_timeout,
                    user_agents=self.config.oddsportal_user_agents,
                )
                provider_name = "OddsPortal"
            elif provider in {"killersports", "ks"}:
                fetcher = KillerSportsFetcher(
                    self.session,
                    base_url=self.config.killersports_base_url,
                    timeout=self.config.closing_odds_timeout,
                    api_key=self.config.killersports_api_key,
                    username=self.config.killersports_username,
                    password=self.config.killersports_password,
                )
                provider_name = "KillerSports"
            else:
                logging.warning("Unknown closing odds provider '%s'", provider)
                return

            archive = fetcher.fetch(seasons)
            if archive.empty:
                logging.warning(
                    "%s did not return any closing odds for seasons %s",
                    provider_name,
                    ", ".join(seasons),
                )
                return

            archive = self._attach_game_ids(archive)
            archive = self._finalize_probabilities(archive)
            updated_rows = self._apply_closing_odds_to_database(archive)
            if updated_rows:
                logging.info(
                    "Applied closing odds to %d games using provider %s",
                    updated_rows,
                    provider_name,
                )
            elif local_provider:
                logging.warning(
                    "Local closing odds data did not match any scheduled games. Verify season/week alignment in %s.",
                    self.config.closing_odds_history_path,
                )
            if local_provider:
                logging.info(
                    "Loaded %d closing odds rows from %s", len(archive), self.config.closing_odds_history_path
                )
            else:
                self._write_history(provider_name, archive)
        finally:
            self.session.close()

    def _attach_game_ids(self, archive: pd.DataFrame) -> pd.DataFrame:
        try:
            games = pd.read_sql_table("nfl_games", self.db.engine)
        except Exception:
            logging.debug("Unable to load nfl_games while attaching closing odds game IDs")
            return archive

        if games.empty:
            return archive

        games["season"] = games["season"].astype(str)
        games["week"] = pd.to_numeric(games["week"], errors="coerce")
        for team_col in ("home_team", "away_team"):
            if team_col in games.columns:
                games[team_col] = games[team_col].apply(normalize_team_abbr)

        merge_cols = [col for col in ("season", "week", "home_team", "away_team") if col in archive.columns]
        if len(merge_cols) < 4:
            return archive

        archive = archive.copy()
        archive["_archive_index"] = np.arange(len(archive))
        archive["season"] = archive["season"].astype(str)
        archive["week"] = pd.to_numeric(archive["week"], errors="coerce")
        for team_col in ("home_team", "away_team"):
            archive[team_col] = archive[team_col].apply(normalize_team_abbr)

        joined = archive.merge(
            games[["game_id", "start_time", "season", "week", "home_team", "away_team"]],
            on=merge_cols,
            how="left",
        )

        if "kickoff_utc" in archive.columns and "game_id" in joined.columns:
            needs_kickoff = joined["game_id"].isna()
            if needs_kickoff.any() and "start_time" in games.columns:
                kickoff_map = archive[["_archive_index", "kickoff_utc"]].copy()
                kickoff_map["kickoff_utc"] = pd.to_datetime(
                    kickoff_map["kickoff_utc"], errors="coerce", utc=True
                )

                games_times = games[["game_id", "start_time"]].copy()
                games_times["start_time"] = pd.to_datetime(
                    games_times["start_time"], errors="coerce", utc=True
                )

                kick_join = kickoff_map.merge(
                    games_times,
                    left_on="kickoff_utc",
                    right_on="start_time",
                    how="left",
                )

                joined = joined.merge(
                    kick_join[["_archive_index", "game_id", "start_time"]],
                    on="_archive_index",
                    how="left",
                    suffixes=("", "_kick"),
                )
                if "game_id_kick" in joined.columns:
                    joined["game_id"] = joined["game_id"].fillna(joined["game_id_kick"])
                    joined.drop(columns=["game_id_kick"], inplace=True)
                if "start_time_kick" in joined.columns:
                    if "game_start" not in joined.columns:
                        joined["game_start"] = pd.NaT
                    joined["game_start"] = joined["game_start"].fillna(
                        joined["start_time_kick"]
                    )
                    joined.drop(columns=["start_time_kick"], inplace=True)

        if "start_time" in joined.columns:
            joined.rename(columns={"start_time": "game_start"}, inplace=True)
        if "_archive_index" in joined.columns:
            joined.drop(columns=["_archive_index"], inplace=True)
        return joined

    @staticmethod
    def _finalize_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for side in ("home", "away"):
            col = f"{side}_closing_moneyline"
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")
                prob_col = f"{side}_closing_implied_prob"
                result[prob_col] = result[col].apply(
                    lambda val: odds_american_to_prob(float(val))
                    if pd.notna(val)
                    else np.nan
                )
        if "closing_line_time" in result.columns:
            result["closing_line_time"] = pd.to_datetime(
                result["closing_line_time"], errors="coerce", utc=True
            )
        if "closing_bookmaker" in result.columns:
            result["closing_bookmaker"] = result["closing_bookmaker"].fillna("")
        return result

    def _apply_closing_odds_to_database(self, archive: pd.DataFrame) -> int:
        if archive is None or archive.empty:
            return 0
        if "game_id" not in archive.columns:
            logging.warning(
                "Closing odds archive missing game_id column; skipping database merge"
            )
            return 0

        working = archive.copy()
        working["game_id"] = working["game_id"].astype(str).str.strip()
        working = working[working["game_id"] != ""]
        if working.empty:
            return 0

        if "closing_line_time" in working.columns:
            working["closing_line_time"] = pd.to_datetime(
                working["closing_line_time"], errors="coerce", utc=True
            )

        sort_cols: List[str] = []
        if "closing_line_time" in working.columns:
            sort_cols.append("closing_line_time")
        if sort_cols:
            working = working.sort_values(sort_cols, ascending=[False] * len(sort_cols))

        working = working.drop_duplicates(subset=["game_id"], keep="first")

        update_columns = [
            "home_closing_moneyline",
            "away_closing_moneyline",
            "home_closing_implied_prob",
            "away_closing_implied_prob",
            "closing_bookmaker",
            "closing_line_time",
        ]
        available_columns = [col for col in update_columns if col in working.columns]
        if not available_columns:
            return 0

        updates: List[Tuple[str, Dict[str, Any]]] = []
        for _, row in working.iterrows():
            payload: Dict[str, Any] = {}
            for col in available_columns:
                value = row.get(col)
                if pd.isna(value):
                    continue
                if col.endswith("moneyline") or col.endswith("implied_prob"):
                    try:
                        payload[col] = float(value)
                    except (TypeError, ValueError):
                        continue
                elif col == "closing_line_time":
                    if isinstance(value, pd.Timestamp):
                        payload[col] = value.to_pydatetime()
                    else:
                        payload[col] = value
                else:
                    text = str(value).strip()
                    if not text:
                        continue
                    payload[col] = text
            if payload:
                game_id = row["game_id"]
                payload["game_id"] = game_id
                updates.append((game_id, payload))

        if not updates:
            return 0

        updated_count = 0
        with self.db.engine.begin() as conn:
            for game_id, payload in updates:
                values = payload.copy()
                values.pop("game_id", None)
                if not values:
                    continue
                stmt = (
                    self.db.games.update()
                    .where(self.db.games.c.game_id == game_id)
                    .values(**values)
                )
                result = conn.execute(stmt)
                updated_count += result.rowcount or 0

        return updated_count

    def _write_history(self, provider_name: str, archive: pd.DataFrame) -> None:
        dest_path = self._history_path()
        try:
            existing = pd.read_csv(dest_path)
        except FileNotFoundError:
            existing = pd.DataFrame()
        except Exception:
            logging.warning("Unable to read existing closing odds history at %s", dest_path)
            existing = pd.DataFrame()

        combined = safe_concat([existing, archive], ignore_index=True)
        if combined.empty:
            logging.warning("No closing odds data available to write after combining frames")
            return

        for team_col in ("home_team", "away_team"):
            if team_col in combined.columns:
                combined[team_col] = combined[team_col].apply(normalize_team_abbr)
        if "closing_line_time" in combined.columns:
            combined["closing_line_time"] = pd.to_datetime(
                combined["closing_line_time"], errors="coerce", utc=True
            )
            combined["closing_line_time"] = combined["closing_line_time"].dt.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            combined["closing_line_time"] = combined["closing_line_time"].replace(
                "NaT", ""
            )

        key_columns = [
            col
            for col in ["game_id", "season", "week", "home_team", "away_team"]
            if col in combined.columns
        ]
        if key_columns:
            combined = combined.drop_duplicates(subset=key_columns, keep="last")

        combined = combined.sort_values(
            [col for col in ("season", "week", "home_team", "away_team") if col in combined.columns]
        )

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(dest_path, index=False)
        logging.info(
            "Wrote %d rows of closing odds to %s using provider %s",
            len(combined),
            dest_path,
            provider_name,
        )

    def _history_path(self) -> Path:
        if self.config.closing_odds_history_path:
            return Path(self.config.closing_odds_history_path)
        default_path = SCRIPT_ROOT / "data" / "closing_odds_history.csv"
        default_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.closing_odds_history_path = str(default_path)
        return default_path


def compute_rmse(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
    """Backwards-compatible RMSE that tolerates older sklearn versions."""

    try:
        return float(mean_squared_error(y_true, y_pred, squared=False))
    except TypeError:
        mse = mean_squared_error(y_true, y_pred)
        return float(np.sqrt(mse))


def _transform_with_feature_names(
    preprocessor: ColumnTransformer,
    frame: pd.DataFrame,
    feature_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Run a ColumnTransformer and ensure a dense DataFrame with aligned columns."""

    if frame is None or frame.empty:
        return pd.DataFrame(index=getattr(frame, "index", None))

    transformed = preprocessor.transform(frame)

    if isinstance(transformed, pd.DataFrame):
        result = transformed.copy()
    else:
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        else:
            transformed = np.asarray(transformed)
        if transformed.ndim == 1:
            transformed = transformed.reshape(-1, 1)
        result = pd.DataFrame(transformed)

    if len(result) == len(frame):
        result.index = frame.index
    else:
        result.index = range(len(result))

    names: Optional[Sequence[str]] = feature_names
    if not names and hasattr(preprocessor, "get_feature_names_out"):
        try:
            names = list(preprocessor.get_feature_names_out())
        except Exception:
            names = None
    if not names or len(names) != result.shape[1]:
        names = [f"feature_{idx}" for idx in range(result.shape[1])]

    result.columns = list(names)
    return result


# === PATCH: Pricing, Calibration & Modeling Utilities ========================
def fair_american(prob: float) -> int:
    """Convert a fair probability to a fair American price."""
    prob = float(np.clip(prob, 1e-6, 1 - 1e-6))
    dec = 1.0 / prob
    american = -100 * (dec - 1) if prob >= 0.5 else 100 * (dec - 1)
    return int(np.round(american))


def american_to_decimal(american: float) -> float:
    return 1 + (100.0 / american if american > 0 else 100.0 / abs(american))


def ev_of_bet(prob: float, american_odds: float, stake: float = 1.0) -> float:
    """Expected value per unit stake against American odds."""
    dec = american_to_decimal(american_odds)
    payout = stake * (dec - 1.0)
    return prob * payout - (1.0 - prob) * stake


def kelly_fraction(prob: float, american_odds: float, max_frac: float = 0.05) -> float:
    """Quarter Kelly on US odds; clamp to sane max."""
    dec = american_to_decimal(american_odds)
    b = dec - 1.0
    p = float(np.clip(prob, 1e-6, 1 - 1e-6))
    q = 1 - p
    k = (b * p - q) / b
    k = max(0.0, k) * 0.25
    return float(min(k, max_frac))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def oof_isotonic(
    model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> Tuple[IsotonicRegression, np.ndarray]:
    """Produce out-of-fold probs and fit isotonic calibration on them."""
    tss = TimeSeriesSplit(n_splits=n_splits)
    oof_pred = np.zeros(len(y), dtype=float)
    for train_idx, valid_idx in tss.split(X):
        clone_model = clone(model)
        clone_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_pred[valid_idx] = clone_model.predict_proba(X.iloc[valid_idx])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_pred, y.values.astype(float))
    return iso, oof_pred


@dataclass
class HurdleTDModel:
    clf: Optional[CalibratedClassifierCV] = None
    reg: Optional[HistGradientBoostingRegressor] = None

    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series, *, n_splits: int = 5
    ) -> None:
        y_bin = (y_train > 0).astype(int)
        base_clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=350)
        self.clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=n_splits)
        self.clf.fit(X_train, y_bin)

        pos_mask = y_train > 0
        X_pos = X_train[pos_mask]
        y_pos = y_train[pos_mask].astype(float)
        if len(y_pos) < 50:
            self.reg = None
            logging.warning("HurdleTDModel: insufficient positives; using μ=1.0 fallback.")
        else:
            self.reg = HistGradientBoostingRegressor(
                max_depth=3, learning_rate=0.06, max_iter=450, loss="poisson"
            )
            self.reg.fit(X_pos, y_pos)

    def pr_anytime(self, X: pd.DataFrame) -> np.ndarray:
        if self.clf is None:
            return np.zeros(len(X))
        return self.clf.predict_proba(X)[:, 1]

    def conditional_mean(self, X: pd.DataFrame) -> np.ndarray:
        if self.reg is None:
            return np.full(len(X), 1.0)
        return np.clip(self.reg.predict(X), 0.2, 3.0)

    def predict_mean(self, X: pd.DataFrame) -> np.ndarray:
        p_any = self.pr_anytime(X)
        mu = self.conditional_mean(X)
        return p_any * mu


@dataclass
class PaperTradeSummary:
    ledger: pd.DataFrame
    window_roi: float
    cumulative_roi: Optional[float]
    graded_bets: int
    closing_coverage: float


class QuantileYards:
    def __init__(self, quantiles: Iterable[float] = (0.1, 0.5, 0.9)):
        self.quantiles = tuple(quantiles)
        self.models: Dict[float, GradientBoostingRegressor] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        X_values = np.asarray(X)
        for quantile in self.quantiles:
            reg = GradientBoostingRegressor(
                loss="quantile",
                alpha=float(quantile),
                max_depth=3,
                n_estimators=350,
                learning_rate=0.05,
            )
            reg.fit(X_values, y)
            self.models[quantile] = reg

    def predict_quantiles(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        X_values = np.asarray(X)
        return {q: model.predict(X_values) for q, model in self.models.items()}

    @staticmethod
    def prob_over(line: float, q_preds: Dict[float, np.ndarray]) -> np.ndarray:
        q10, q50, q90 = q_preds[0.1], q_preds[0.5], q_preds[0.9]
        p_le = np.where(
            line <= q10,
            0.10,
            np.where(
                line <= q50,
                0.50
                - 0.40
                * (line - q10)
                / np.clip(q50 - q10, 1e-3, None),
                np.where(
                    line <= q90,
                    0.90
                    - 0.40
                    * (line - q50)
                    / np.clip(q90 - q50, 1e-3, None),
                    0.95,
                ),
            ),
        )
        return np.clip(1.0 - p_le, 0.01, 0.99)


class TeamPoissonTotals:
    def __init__(self, alpha: float = 1.0):
        self.home = PoissonRegressor(alpha=alpha, max_iter=600)
        self.away = PoissonRegressor(alpha=alpha, max_iter=600)

    def fit(
        self,
        X_home: pd.DataFrame,
        y_home: pd.Series,
        X_away: pd.DataFrame,
        y_away: pd.Series,
    ) -> None:
        self.home.fit(X_home, y_home)
        self.away.fit(X_away, y_away)

    def predict_lambda(
        self, X_home: pd.DataFrame, X_away: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        lambda_home = np.clip(self.home.predict(X_home), 0.1, 60)
        lambda_away = np.clip(self.away.predict(X_away), 0.1, 60)
        return lambda_home, lambda_away

    @staticmethod
    def _poisson_cdf(rate: float, k: int) -> float:
        """Compute cumulative probability P(X <= k) for X~Poisson(rate)."""

        if k < 0:
            return 0.0
        if rate <= 0.0:
            return 1.0

        term = math.exp(-rate)
        cumulative = term
        for i in range(1, k + 1):
            term *= rate / i
            cumulative += term
        return float(min(1.0, cumulative))

    @staticmethod
    def prob_total_over(
        lambda_home: np.ndarray,
        lambda_away: np.ndarray,
        total: float,
    ) -> np.ndarray:
        """Exact probability the combined score exceeds the listed total."""

        lam_home = np.asarray(lambda_home, dtype=float)
        lam_away = np.asarray(lambda_away, dtype=float)
        combined = lam_home + lam_away

        totals = np.broadcast_to(np.asarray(total, dtype=float), combined.shape).astype(float)
        flat_combined = combined.reshape(-1)
        flat_totals = totals.reshape(-1)

        probs = np.empty_like(flat_combined)
        for idx, (lam, line) in enumerate(zip(flat_combined, flat_totals)):
            threshold = math.floor(line)
            cdf_val = TeamPoissonTotals._poisson_cdf(lam, threshold)
            probs[idx] = max(0.0, min(1.0, 1.0 - cdf_val))

        return probs.reshape(combined.shape)

    @staticmethod
    def _home_win_probability_pair(lam_h: float, lam_a: float, tol: float = 1e-10) -> float:
        if lam_h < 0 or lam_a < 0:
            return 0.5

        pmf_home = math.exp(-lam_h)
        pmf_away = math.exp(-lam_a)
        cdf_home = pmf_home

        prob_home_win = 0.0
        prob_tie = pmf_home * pmf_away
        cdf_away_prev = pmf_away

        max_rate = max(lam_h, lam_a)
        max_iter = max(50, int(math.ceil(max_rate + 10.0 * math.sqrt(max_rate + 1.0))))

        for k in range(1, max_iter + 1):
            pmf_home *= lam_h / k if lam_h > 0 else 0.0
            pmf_away *= lam_a / k if lam_a > 0 else 0.0

            prob_home_win += pmf_home * cdf_away_prev
            prob_tie += pmf_home * pmf_away

            cdf_home = min(1.0, cdf_home + pmf_home)
            cdf_away_prev = min(1.0, cdf_away_prev + pmf_away)

            if (1.0 - cdf_home) < tol and (1.0 - cdf_away_prev) < tol:
                break

        if cdf_home < 1.0:
            prob_home_win += (1.0 - cdf_home) * cdf_away_prev

        probability = prob_home_win + 0.5 * prob_tie
        return float(min(1.0, max(0.0, probability)))

    @staticmethod
    def win_probability(
        lambda_home: np.ndarray,
        lambda_away: np.ndarray,
    ) -> np.ndarray:
        """Deterministic estimate of home win probability from Poisson rates."""

        lam_home = np.asarray(lambda_home, dtype=float)
        lam_away = np.asarray(lambda_away, dtype=float)

        flat_home = lam_home.reshape(-1)
        flat_away = lam_away.reshape(-1)
        probs = np.empty_like(flat_home)

        for idx, (lh, la) in enumerate(zip(flat_home, flat_away)):
            probs[idx] = TeamPoissonTotals._home_win_probability_pair(lh, la)

        return probs.reshape(lam_home.shape)


def pick_best_odds(odds_df: pd.DataFrame, by_cols: Iterable[str], price_col: str) -> pd.DataFrame:
    out = (
        odds_df.sort_values(
            by=list(by_cols) + [price_col],
            ascending=[True] * len(list(by_cols)) + [False],
        ).drop_duplicates(subset=list(by_cols), keep="first")
    )
    return out


def confidence_bucket(ev: float, prob: float) -> str:
    if ev >= 0.05 and prob >= 0.60:
        return "A"
    if ev >= 0.03 and prob >= 0.55:
        return "B"
    if ev >= 0.02 and prob >= 0.53:
        return "C"
    return "Pass"


EV_MIN_PROPS = 0.025
EV_MIN_TOTALS = 0.020
EV_MIN_SIDES = 0.020
CONF_ECE_MAX = 0.035


def filter_ev(df: pd.DataFrame, min_ev: float) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df[df["ev"] >= min_ev].copy()


def write_csv_safely(df: pd.DataFrame, path: str) -> None:
    try:
        if df is None or df.empty:
            logging.info("No rows to write for %s", path)
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logging.info("Wrote %d rows -> %s", len(df), path)
    except Exception:
        logging.exception("Failed to write %s", path)


def append_csv_safely(df: pd.DataFrame, path: str) -> None:
    """Append rows to a CSV, creating it if necessary."""

    if df is None or df.empty:
        logging.info("No rows to append for %s", path)
        return

    try:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        exists = path_obj.exists()
        df.to_csv(path_obj, mode="a" if exists else "w", header=not exists, index=False)
        logging.info("Appended %d rows -> %s", len(df), path)
    except Exception:
        logging.exception("Failed to append %s", path)


def extract_pricing_odds(
    odds_payload: Iterable[Dict[str, Any]],
    valid_game_ids: Optional[Iterable[Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert bookmaker payload into flat tables for props and totals pricing."""

    if valid_game_ids is None:
        valid_ids: Optional[Set[str]] = None
    else:
        valid_ids = {str(gid) for gid in valid_game_ids if gid is not None}

    player_rows: List[Dict[str, Any]] = []
    total_rows: List[Dict[str, Any]] = []

    for event in odds_payload or []:
        game_id = str(event.get("id") or "")
        if not game_id:
            continue
        if valid_ids is not None and game_id not in valid_ids:
            continue

        teams = [team for team in (event.get("teams") or []) if team]
        home_raw = event.get("home_team") or (teams[0] if teams else None)
        away_raw = event.get("away_team")
        if not away_raw and teams:
            away_raw = next((team for team in teams if team != home_raw), teams[0])

        home_team = normalize_team_abbr(home_raw)
        away_team = normalize_team_abbr(away_raw)

        bookmakers = event.get("bookmakers", [])
        for bookmaker in bookmakers:
            sportsbook = bookmaker.get("key") or bookmaker.get("title") or "unknown"
            last_update = parse_dt(bookmaker.get("last_update"))
            for market in bookmaker.get("markets", []):
                key = (market.get("key") or "").lower()
                outcomes = market.get("outcomes", []) or []

                if key == "totals":
                    for outcome in outcomes:
                        side = (outcome.get("name") or "").title()
                        total_line = outcome.get("point")
                        price = outcome.get("price")
                        if side not in {"Over", "Under"}:
                            continue
                        if price is None or total_line is None:
                            continue
                        try:
                            total_value = float(total_line)
                        except (TypeError, ValueError):
                            continue
                        try:
                            american_price = float(price)
                        except (TypeError, ValueError):
                            continue
                        total_rows.append(
                            {
                                "market": "total",
                                "game_id": game_id,
                                "away_team": away_team,
                                "home_team": home_team,
                                "side": side,
                                "total": total_value,
                                "american_odds": american_price,
                                "sportsbook": sportsbook,
                                "event_id": game_id,
                                "last_update": last_update,
                            }
                        )

                if key in PLAYER_PROP_MARKET_COLUMN_MAP:
                    stat_key = PLAYER_PROP_MARKET_COLUMN_MAP[key].replace("line_", "")
                    player_buckets: Dict[str, Dict[str, Any]] = {}
                    for outcome in outcomes:
                        name_raw = str(outcome.get("name") or "").strip()
                        desc_raw = str(outcome.get("description") or "").strip()
                        participant = str(
                            outcome.get("participant")
                            or outcome.get("player")
                            or outcome.get("player_name")
                            or ""
                        ).strip()
                        side = None
                        player_name = None

                        name_lower = name_raw.lower()
                        desc_lower = desc_raw.lower()
                        side_tokens = {"over", "under", "yes", "no"}

                        if name_lower in side_tokens:
                            side = name_raw.title()
                            player_name = desc_raw or participant
                        elif desc_lower in side_tokens:
                            side = desc_raw.title()
                            player_name = name_raw or participant
                        elif (
                            name_lower.endswith(" over")
                            or name_lower.endswith(" under")
                            or name_lower.endswith(" yes")
                            or name_lower.endswith(" no")
                        ):
                            tokens = name_lower.rsplit(" ", 1)
                            side = tokens[1].title()
                            player_name = name_raw[: -len(tokens[1])].strip()
                        elif participant:
                            player_name = participant

                        if not player_name or side is None:
                            continue

                        player_line = outcome.get("line") or outcome.get("point")
                        try:
                            line_value = float(player_line) if player_line is not None else None
                        except (TypeError, ValueError):
                            line_value = None
                        try:
                            price_value = float(outcome.get("price")) if outcome.get("price") is not None else None
                        except (TypeError, ValueError):
                            price_value = None

                        if line_value is None or price_value is None:
                            continue

                        team_abbr = normalize_team_abbr(outcome.get("team"))
                        player_key = robust_player_name_key(player_name)
                        bucket = player_buckets.setdefault(
                            player_key,
                            {
                                "player_name": player_name,
                                "player_id": outcome.get("player_id")
                                or outcome.get("participant_id")
                                or outcome.get("id"),
                                "team": team_abbr,
                                "line": line_value,
                                "over": None,
                                "under": None,
                            },
                        )
                        bucket["line"] = line_value
                        if side == "Over":
                            bucket["over"] = price_value
                        elif side == "Under":
                            bucket["under"] = price_value

                    for player_key, info in player_buckets.items():
                        if info.get("line") is None:
                            continue
                        player_identifier = info.get("player_id")
                        if player_identifier is None:
                            player_identifier = player_key
                        team_abbr = info.get("team")
                        opponent_abbr = None
                        if team_abbr:
                            if team_abbr == home_team:
                                opponent_abbr = away_team
                            elif team_abbr == away_team:
                                opponent_abbr = home_team
                        if info.get("over") is not None:
                            player_rows.append(
                                {
                                    "market": stat_key,
                                    "player_id": player_identifier,
                                    "player_name": info.get("player_name"),
                                    "team": team_abbr,
                                    "opponent": opponent_abbr,
                                    "line": info.get("line"),
                                    "american_odds": info.get("over"),
                                    "side": "Over",
                                    "sportsbook": sportsbook,
                                    "event_id": game_id,
                                    "game_id": game_id,
                                    "last_update": last_update,
                                }
                            )
                        if info.get("under") is not None:
                            player_rows.append(
                                {
                                    "market": stat_key,
                                    "player_id": player_identifier,
                                    "player_name": info.get("player_name"),
                                    "team": team_abbr,
                                    "opponent": opponent_abbr,
                                    "line": info.get("line"),
                                    "american_odds": info.get("under"),
                                    "side": "Under",
                                    "sportsbook": sportsbook,
                                    "event_id": game_id,
                                    "game_id": game_id,
                                    "last_update": last_update,
                                }
                            )

    odds_players = pd.DataFrame(player_rows)
    odds_totals = pd.DataFrame(total_rows)

    if valid_ids is not None and not odds_totals.empty:
        odds_totals = odds_totals[odds_totals["game_id"].astype(str).isin(valid_ids)]

    return odds_players, odds_totals


def pick_allowed_positions(target: str) -> Optional[Set[str]]:
    return TARGET_ALLOWED_POSITIONS.get(target)


# =============================================================================


PLAYER_PROP_ALLOWED_SIDES: Dict[str, Set[str]] = {
    "anytime_td": {"yes", "over"},
    "passing_yards": {"over"},
    "receiving_yards": {"over"},
    "receptions": {"over"},
    "rushing_yards": {"over"},
}


def _merge_player_prop_on_key(
    preds: pd.DataFrame,
    offers: pd.DataFrame,
    key_col: str,
    allowed_side_map: Optional[Dict[str, Set[str]]] = None,
) -> pd.DataFrame:
    """Merge prediction rows with sportsbook offers using a specific join key."""

    if preds.empty or offers.empty:
        return pd.DataFrame()

    allowed_lookup = allowed_side_map or PLAYER_PROP_ALLOWED_SIDES

    key_cols: List[str] = ["market", key_col]

    # Include event-level context when available so duplicate matchups resolve cleanly.
    if not key_col.endswith("_event_key"):
        if "_event_key" in preds.columns and "_event_key" in offers.columns:
            if preds["_event_key"].astype(bool).any() and offers["_event_key"].astype(bool).any():
                key_cols.append("_event_key")

    if "line" in offers.columns:
        key_cols.append("line")
    if "side" in offers.columns:
        key_cols.append("side")

    best = pick_best_odds(offers, by_cols=key_cols, price_col="american_odds")
    if best.empty:
        return pd.DataFrame()

    if "side" in best.columns:
        best = best.copy()
        best["_side_norm"] = best["side"].fillna("").str.lower()
        best = best[
            best.apply(
                lambda row: row["_side_norm"]
                in allowed_lookup.get(row["market"], {row["_side_norm"]}),
                axis=1,
            )
        ].drop(columns=["_side_norm"], errors="ignore")
        if best.empty:
            return pd.DataFrame()

    join_cols = [col for col in key_cols if col in preds.columns and col in best.columns]
    if not join_cols:
        join_cols = ["market", key_col]

    return preds.merge(best, on=join_cols, how="inner", suffixes=("", "_book"))


def build_player_prop_candidates(
    pred_df: pd.DataFrame, odds_df: pd.DataFrame
) -> pd.DataFrame:
    if pred_df.empty or odds_df.empty:
        return pd.DataFrame()

    pred_df = pred_df.copy().reset_index(drop=True)
    odds_df = odds_df.copy().reset_index(drop=True)

    def _normalize_identifier(value: Any) -> str:
        if pd.isna(value):
            return ""
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"}:
            return ""
        if text.endswith(".0"):
            text = text[:-2]
        return text

    def _extract_keys(row: pd.Series) -> pd.Series:
        identifier = _normalize_identifier(row.get("player_id"))
        name = row.get("player_name") or row.get("player") or row.get("name")
        if isinstance(name, str) and name.strip():
            name_key = robust_player_name_key(name)
        else:
            name_key = ""

        team_val = row.get("team") or row.get("team_abbr")
        team_norm = normalize_team_abbr(team_val)
        if team_norm:
            team_key = team_norm
        elif isinstance(team_val, str):
            team_key = team_val.strip().upper()
        else:
            team_key = ""

        team_join = f"{name_key}::{team_key}" if name_key and team_key else name_key

        event_val = row.get("event_id") or row.get("game_id")
        event_key = _normalize_identifier(event_val)

        def _with_event(base: str) -> str:
            if base and event_key:
                return f"{base}::{event_key}"
            return ""

        return pd.Series(
            {
                "_event_key": event_key,
                "_player_id_key": identifier,
                "_player_name_key": name_key,
                "_player_team_key": team_join,
                "_player_id_event_key": _with_event(identifier),
                "_player_name_event_key": _with_event(name_key),
                "_player_team_event_key": _with_event(team_join if team_join else name_key),
            }
        )

    key_columns = [
        "_event_key",
        "_player_id_key",
        "_player_name_key",
        "_player_team_key",
        "_player_id_event_key",
        "_player_name_event_key",
        "_player_team_event_key",
    ]

    def _as_series(df: pd.DataFrame, column: str) -> pd.Series:
        """Return the named column as a Series even if duplicates created a DataFrame."""

        values = df[column]
        if isinstance(values, pd.DataFrame):
            # Retain the first occurrence – duplicate columns are equivalent for our keys.
            values = values.iloc[:, 0]
        return values

    def _annotate_keys(frame: pd.DataFrame) -> pd.DataFrame:
        base = frame.copy()
        base = base.loc[:, ~base.columns.duplicated(keep="first")]
        base = base.drop(columns=key_columns, errors="ignore")

        if base.empty:
            result = base
            for col in key_columns:
                result[col] = pd.Series(dtype=str)
            return result

        keys = frame.apply(_extract_keys, axis=1)
        result = base
        for col in key_columns:
            result[col] = keys[col]

        team_values = _as_series(result, "_player_team_key")
        mask = team_values.astype(bool)
        if not mask.all():
            name_values = _as_series(result, "_player_name_key")
            replacement = name_values.loc[~mask]
            result.loc[~mask, "_player_team_key"] = replacement

        team_event_values = _as_series(result, "_player_team_event_key")
        mask_event = team_event_values.astype(bool)
        if not mask_event.all():
            name_event_values = _as_series(result, "_player_name_event_key")
            replacement_event = name_event_values.loc[~mask_event]
            result.loc[~mask_event, "_player_team_event_key"] = replacement_event
        return result

    pred_df = _annotate_keys(pred_df)
    odds_df = _annotate_keys(odds_df)

    pred_df = pred_df[
        pred_df["_player_id_key"].astype(bool)
        | pred_df["_player_team_key"].astype(bool)
        | pred_df["_player_name_key"].astype(bool)
    ].copy()
    odds_df = odds_df[
        odds_df["_player_id_key"].astype(bool)
        | odds_df["_player_team_key"].astype(bool)
        | odds_df["_player_name_key"].astype(bool)
    ].copy()

    def _as_series(df: pd.DataFrame, column: str) -> pd.Series:
        """Return the named column as a Series even if duplicates created a DataFrame."""

        values = df[column]
        if isinstance(values, pd.DataFrame):
            # Retain the first occurrence – duplicate columns are equivalent for our keys.
            values = values.iloc[:, 0]
        return values

    def _annotate_keys(frame: pd.DataFrame) -> pd.DataFrame:
        base = frame.copy()
        base = base.loc[:, ~base.columns.duplicated(keep="first")]
        base = base.drop(columns=key_columns, errors="ignore")

        if base.empty:
            result = base
            for col in key_columns:
                result[col] = pd.Series(dtype=str)
            return result

        keys = frame.apply(_extract_keys, axis=1)
        result = base
        for col in key_columns:
            result[col] = keys[col]

        team_values = _as_series(result, "_player_team_key")
        mask = team_values.astype(bool)
        if not mask.all():
            name_values = _as_series(result, "_player_name_key")
            replacement = name_values.loc[~mask]
            result.loc[~mask, "_player_team_key"] = replacement

        team_event_values = _as_series(result, "_player_team_event_key")
        mask_event = team_event_values.astype(bool)
        if not mask_event.all():
            name_event_values = _as_series(result, "_player_name_event_key")
            replacement_event = name_event_values.loc[~mask_event]
            result.loc[~mask_event, "_player_team_event_key"] = replacement_event
        return result

    pred_df = _annotate_keys(pred_df)
    odds_df = _annotate_keys(odds_df)

    pred_df = pred_df[
        pred_df["_player_id_key"].astype(bool)
        | pred_df["_player_team_key"].astype(bool)
        | pred_df["_player_name_key"].astype(bool)
    ].copy()
    odds_df = odds_df[
        odds_df["_player_id_key"].astype(bool)
        | odds_df["_player_team_key"].astype(bool)
        | odds_df["_player_name_key"].astype(bool)
    ].copy()

    pred_df["_pred_index"] = np.arange(len(pred_df))
    odds_df["_odds_index"] = np.arange(len(odds_df))
    pred_df = pred_df.set_index("_pred_index", drop=False)
    odds_df = odds_df.set_index("_odds_index", drop=False)

    merged_frames: List[pd.DataFrame] = []
    remaining_pred = pred_df
    remaining_odds = odds_df

    for key_col in (
        "_player_id_event_key",
        "_player_team_event_key",
        "_player_name_event_key",
        "_player_id_key",
        "_player_team_key",
        "_player_name_key",
    ):
        preds_slice = remaining_pred[remaining_pred[key_col].astype(bool)].copy()
        offers_slice = remaining_odds[remaining_odds[key_col].astype(bool)].copy()
        merged_slice = _merge_player_prop_on_key(
            preds_slice, offers_slice, key_col
        )
        if merged_slice.empty:
            continue
        merged_frames.append(merged_slice)
        matched_pred_idx = merged_slice["_pred_index"].unique().tolist()
        matched_odds_idx = (
            merged_slice["_odds_index_book"].unique().tolist()
            if "_odds_index_book" in merged_slice.columns
            else []
        )
        if matched_pred_idx:
            remaining_pred = remaining_pred.drop(index=matched_pred_idx, errors="ignore")
        if matched_odds_idx:
            remaining_odds = remaining_odds.drop(index=matched_odds_idx, errors="ignore")

    pred_df["_pred_index"] = np.arange(len(pred_df))
    odds_df["_odds_index"] = np.arange(len(odds_df))
    pred_df = pred_df.set_index("_pred_index", drop=False)
    odds_df = odds_df.set_index("_odds_index", drop=False)

    merged_frames: List[pd.DataFrame] = []
    remaining_pred = pred_df
    remaining_odds = odds_df

    for key_col in (
        "_player_id_event_key",
        "_player_team_event_key",
        "_player_name_event_key",
        "_player_id_key",
        "_player_team_key",
        "_player_name_key",
    ):
        preds_slice = remaining_pred[remaining_pred[key_col].astype(bool)].copy()
        offers_slice = remaining_odds[remaining_odds[key_col].astype(bool)].copy()
        merged_slice = _merge_player_prop_on_key(
            preds_slice, offers_slice, key_col
        )
        if merged_slice.empty:
            continue
        merged_frames.append(merged_slice)
        matched_pred_idx = merged_slice["_pred_index"].unique().tolist()
        matched_odds_idx = (
            merged_slice["_odds_index_book"].unique().tolist()
            if "_odds_index_book" in merged_slice.columns
            else []
        )
        if matched_pred_idx:
            remaining_pred = remaining_pred.drop(index=matched_pred_idx, errors="ignore")
        if matched_odds_idx:
            remaining_odds = remaining_odds.drop(index=matched_odds_idx, errors="ignore")

    pred_df["_pred_index"] = np.arange(len(pred_df))
    odds_df["_odds_index"] = np.arange(len(odds_df))
    pred_df = pred_df.set_index("_pred_index", drop=False)
    odds_df = odds_df.set_index("_odds_index", drop=False)

    merged_frames: List[pd.DataFrame] = []
    remaining_pred = pred_df
    remaining_odds = odds_df

    for key_col in (
        "_player_id_event_key",
        "_player_team_event_key",
        "_player_name_event_key",
        "_player_id_key",
        "_player_team_key",
        "_player_name_key",
    ):
        preds_slice = remaining_pred[remaining_pred[key_col].astype(bool)].copy()
        offers_slice = remaining_odds[remaining_odds[key_col].astype(bool)].copy()
        merged_slice = _merge_player_prop_on_key(
            preds_slice, offers_slice, key_col
        )
        if merged_slice.empty:
            continue
        merged_frames.append(merged_slice)
        matched_pred_idx = merged_slice["_pred_index"].unique().tolist()
        matched_odds_idx = (
            merged_slice["_odds_index_book"].unique().tolist()
            if "_odds_index_book" in merged_slice.columns
            else []
        )
        if matched_pred_idx:
            remaining_pred = remaining_pred.drop(index=matched_pred_idx, errors="ignore")
        if matched_odds_idx:
            remaining_odds = remaining_odds.drop(index=matched_odds_idx, errors="ignore")

    if not merged_frames:
        logging.info(
            "Player prop odds merge produced 0 matches (pred_rows=%d, odds_rows=%d)",
            len(pred_df),
            len(odds_df),
        )
        return pd.DataFrame()

    merged = pd.concat(merged_frames, ignore_index=True, sort=False)
    logging.info(
        "Player prop odds merge matched %d predictions to %d sportsbook offers (pred_rows=%d, odds_rows=%d)",
        merged["_pred_index"].nunique(),
        merged.get("_odds_index_book", merged.get("_odds_index", pd.Series())).nunique(),
        len(pred_df),
        len(odds_df),
    )
    rows: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        market = row["market"]
        american = float(row.get("american_odds", np.nan))
        if market in {"receiving_yards", "passing_yards", "receptions"}:
            line = float(row.get("line", np.nan))
            quantiles = {
                0.1: np.array([float(row.get("q10", np.nan))]),
                0.5: np.array([float(row.get("pred_median", np.nan))]),
                0.9: np.array([float(row.get("q90", np.nan))]),
            }
            if np.isnan(line) or any(np.isnan(vals[0]) for vals in quantiles.values()):
                continue
            prob_over = float(QuantileYards.prob_over(line, quantiles)[0])
            ev = ev_of_bet(prob_over, american)
            rows.append(
                {
                    "market": market,
                    "player_id": row.get("player_id"),
                    "player": row.get("player_name"),
                    "team": row.get("team"),
                    "opp": row.get("opponent"),
                    "side": "Over",
                    "line": line,
                    "fair_prob": prob_over,
                    "fair_american": fair_american(prob_over),
                    "best_american": american,
                    "ev": ev,
                    "kelly_quarter": kelly_fraction(prob_over, american),
                }
            )
        elif market == "anytime_td":
            prob_any = float(row.get("anytime_prob", np.nan))
            if np.isnan(prob_any):
                continue
            ev = ev_of_bet(prob_any, american)
            rows.append(
                {
                    "market": market,
                    "player_id": row.get("player_id"),
                    "player": row.get("player_name"),
                    "team": row.get("team"),
                    "opp": row.get("opponent"),
                    "side": "Yes",
                    "line": np.nan,
                    "fair_prob": prob_any,
                    "fair_american": fair_american(prob_any),
                    "best_american": american,
                    "ev": ev,
                    "kelly_quarter": kelly_fraction(prob_any, american),
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["confidence"] = [
            confidence_bucket(ev, prob) for ev, prob in zip(result["ev"], result["fair_prob"])
        ]
        result = result.sort_values(["confidence", "ev"], ascending=[True, False])
    return result


def build_game_totals_candidates(
    week_games_df: pd.DataFrame,
    feats_home: pd.DataFrame,
    feats_away: pd.DataFrame,
    odds_df: pd.DataFrame,
    tpois: TeamPoissonTotals,
) -> pd.DataFrame:
    if odds_df.empty or week_games_df.empty:
        return pd.DataFrame()
    offers = odds_df.query("market == 'total'").copy()
    if offers.empty:
        return pd.DataFrame()
    offers["game_id"] = offers["game_id"].astype(str)

    lam_home, lam_away = tpois.predict_lambda(feats_home, feats_away)
    model_total = lam_home + lam_away
    rows: List[Dict[str, Any]] = []
    for idx, game in week_games_df.reset_index(drop=True).iterrows():
        gid = str(game.get("game_id"))
        offers_game = offers[offers["game_id"] == gid]
        if offers_game.empty:
            continue
        for _, offer in offers_game.iterrows():
            side = offer.get("side")
            line = float(offer.get("total", np.nan))
            american = float(offer.get("american_odds", np.nan))
            prob_over = float(
                TeamPoissonTotals.prob_total_over(
                    np.array([lam_home[idx]]), np.array([lam_away[idx]]), line
                )[0]
            )
            prob = prob_over if str(side).lower() == "over" else 1.0 - prob_over
            ev_val = ev_of_bet(prob, american)
            rows.append(
                {
                    "market": "total",
                    "game_id": gid,
                    "away": game.get("away_team"),
                    "home": game.get("home_team"),
                    "side": side,
                    "line": line,
                    "fair_prob": prob,
                    "fair_american": fair_american(prob),
                    "best_american": american,
                    "ev": ev_val,
                    "kelly_quarter": kelly_fraction(prob, american),
                    "model_total": float(model_total[idx]),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["confidence"] = [
            confidence_bucket(ev, prob) for ev, prob in zip(out["ev"], out["fair_prob"])
        ]
        out = out.sort_values(["confidence", "ev"], ascending=[True, False])
    return out


def emit_priced_picks(
    week_key: str,
    player_pred_tables: Dict[str, pd.DataFrame],
    odds_players: pd.DataFrame,
    week_games_df: pd.DataFrame,
    feats_home: pd.DataFrame,
    feats_away: pd.DataFrame,
    odds_games: pd.DataFrame,
    tpois: Optional[TeamPoissonTotals],
    out_dir: Path,
) -> Dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)

    stacked = []
    fallback_tables: List[pd.DataFrame] = []
    for market, table in player_pred_tables.items():
        if table is None or table.empty:
            continue
        table = table.copy()
        table["market"] = market
        stacked.append(table)

        fallback = table.copy()
        if market == "anytime_td":
            fallback["model_probability"] = fallback.get("anytime_prob")
            fallback["recommended_line"] = np.nan
            fallback["range_low"] = np.nan
            fallback["range_high"] = np.nan
        else:
            fallback["model_probability"] = np.nan
            fallback["recommended_line"] = fallback.get("pred_median")
            fallback["range_low"] = fallback.get("q10")
            fallback["range_high"] = fallback.get("q90")
        fallback_tables.append(fallback)
    preds_all = pd.concat(stacked, ignore_index=True) if stacked else pd.DataFrame()

    props_priced = build_player_prop_candidates(preds_all, odds_players) if not preds_all.empty else pd.DataFrame()
    props_filtered = filter_ev(props_priced, EV_MIN_PROPS)
    write_csv_safely(props_filtered, str(out_dir / f"player_props_priced_{week_key}.csv"))

    model_props = pd.DataFrame()
    if fallback_tables:
        model_props = pd.concat(fallback_tables, ignore_index=True)
        columns_order = [
            col
            for col in [
                "market",
                "player_id",
                "player_name",
                "team",
                "opponent",
                "position",
                "recommended_line",
                "range_low",
                "range_high",
                "model_probability",
            ]
            if col in model_props.columns
        ]
        model_props = model_props.loc[:, columns_order]
        model_props_path = out_dir / f"player_props_model_{week_key}.csv"
        write_csv_safely(model_props, str(model_props_path))
        if not model_props.empty:
            logging.info(
                "Saved %d model-only prop forecasts to %s",
                len(model_props),
                model_props_path,
            )

    totals_priced = pd.DataFrame()
    if tpois is not None and not feats_home.empty and not odds_games.empty:
        totals_priced = build_game_totals_candidates(
            week_games_df=week_games_df,
            feats_home=feats_home,
            feats_away=feats_away,
            odds_df=odds_games,
            tpois=tpois,
        )
        totals_filtered = filter_ev(totals_priced, EV_MIN_TOTALS)
        write_csv_safely(totals_filtered, str(out_dir / f"game_totals_priced_{week_key}.csv"))
    else:
        totals_filtered = pd.DataFrame()

    model_totals = pd.DataFrame()
    if tpois is not None and not feats_home.empty:
        lam_home, lam_away = tpois.predict_lambda(feats_home, feats_away)
        model_totals = week_games_df.copy()
        model_totals = model_totals.assign(
            model_home_lambda=lam_home,
            model_away_lambda=lam_away,
            model_total=lam_home + lam_away,
            model_spread=lam_home - lam_away,
            model_home_win_prob=TeamPoissonTotals.win_probability(lam_home, lam_away),
        )
        model_totals_path = out_dir / f"game_totals_model_{week_key}.csv"
        write_csv_safely(model_totals, str(model_totals_path))
        if not model_totals.empty:
            logging.info(
                "Saved %d model-only game total forecasts to %s",
                len(model_totals),
                model_totals_path,
            )

    paper_rows: List[Dict[str, Any]] = []
    run_timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    if props_filtered is not None and not props_filtered.empty:
        for row in props_filtered.to_dict("records"):
            paper_rows.append(
                {
                    "timestamp_utc": run_timestamp,
                    "week_key": week_key,
                    "bet_type": "player_prop",
                    "market": row.get("market"),
                    "player": row.get("player"),
                    "team": row.get("team"),
                    "opponent": row.get("opp"),
                    "side": row.get("side"),
                    "line": row.get("line"),
                    "best_american": row.get("best_american"),
                    "fair_american": row.get("fair_american"),
                    "ev": row.get("ev"),
                    "confidence": row.get("confidence"),
                    "kelly_quarter": row.get("kelly_quarter"),
                    "event_id": row.get("event_id") or row.get("game_id"),
                }
            )

    if totals_filtered is not None and not totals_filtered.empty:
        for row in totals_filtered.to_dict("records"):
            paper_rows.append(
                {
                    "timestamp_utc": run_timestamp,
                    "week_key": week_key,
                    "bet_type": "game_total",
                    "market": "total",
                    "away_team": row.get("away"),
                    "home_team": row.get("home"),
                    "side": row.get("side"),
                    "line": row.get("line"),
                    "best_american": row.get("best_american"),
                    "fair_american": row.get("fair_american"),
                    "ev": row.get("ev"),
                    "confidence": row.get("confidence"),
                    "kelly_quarter": row.get("kelly_quarter"),
                    "model_total": row.get("model_total"),
                    "event_id": row.get("event_id") or row.get("game_id"),
                }
            )

    if paper_rows:
        paper_log_path = out_dir / "paper_trading_log.csv"
        append_csv_safely(pd.DataFrame(paper_rows), str(paper_log_path))

    try:
        if props_filtered is not None and not props_filtered.empty:
            logging.info(
                "Top player props:\n%s",
                props_filtered.sort_values("ev", ascending=False)
                .head(12)[
                    [
                        "market",
                        "player",
                        "team",
                        "opp",
                        "side",
                        "line",
                        "best_american",
                        "fair_american",
                        "ev",
                        "confidence",
                        "kelly_quarter",
                    ]
                ]
                .to_string(index=False),
            )
        if totals_filtered is not None and not totals_filtered.empty:
            logging.info(
                "Top totals:\n%s",
                totals_filtered.sort_values("ev", ascending=False)
                .head(8)[
                    [
                        "away",
                        "home",
                        "side",
                        "line",
                        "best_american",
                        "fair_american",
                        "ev",
                        "model_total",
                        "kelly_quarter",
                        "confidence",
                    ]
                ]
                .to_string(index=False),
            )
    except Exception:
        logging.debug("Failed to print priced pick summary", exc_info=True)

    return {
        "props": props_filtered if props_filtered is not None else pd.DataFrame(),
        "totals": totals_filtered if totals_filtered is not None else pd.DataFrame(),
        "model_props": model_props,
        "model_totals": model_totals,
    }


def make_quantile_pred_table(
    qmodel: QuantileYards,
    preprocessor: ColumnTransformer,
    feature_names: Optional[Sequence[str]],
    X_week: pd.DataFrame,
    player_index: pd.DataFrame,
    market_name: str,
) -> pd.DataFrame:
    if X_week.empty:
        return pd.DataFrame()

    processed = _transform_with_feature_names(preprocessor, X_week, feature_names)
    preds = qmodel.predict_quantiles(processed)
    output = player_index.copy()
    output["q10"] = preds[0.1]
    output["pred_median"] = preds[0.5]
    output["q90"] = preds[0.9]
    output["market"] = market_name
    return output


def make_anytime_td_table(
    hmodel: HurdleTDModel,
    preprocessor: ColumnTransformer,
    feature_names: Optional[Sequence[str]],
    X_week: pd.DataFrame,
    player_index: pd.DataFrame,
) -> pd.DataFrame:
    if X_week.empty:
        return pd.DataFrame()
    processed = _transform_with_feature_names(preprocessor, X_week, feature_names)
    output = player_index.copy()
    output["anytime_prob"] = hmodel.pr_anytime(processed)
    output["market"] = "anytime_td"
    return output


def compute_recency_usage_weights(frame: pd.DataFrame) -> pd.Series:
    """Compute recency- and usage-based weights for player rows."""

    if frame is None or frame.empty:
        return pd.Series(dtype=float, index=getattr(frame, "index", None))

    recency_cols = [
        "start_time",
        "local_start_time",
        "game_datetime",
        "game_date",
        "kickoff",
    ]
    recency_weight = pd.Series(1.0, index=frame.index, dtype=float)
    found_time = False
    for col in recency_cols:
        if col in frame.columns:
            candidate = pd.to_datetime(frame[col], errors="coerce")
            if candidate.notna().any():
                latest = candidate.max()
                age_days = (latest - candidate).dt.total_seconds() / 86400.0
                age_days = age_days.fillna(age_days.max() or 0.0)
                halflife_days = 21.0
                recency_weight = np.exp(-age_days / halflife_days)
                found_time = True
                break
    if not found_time and {"season", "week"}.issubset(frame.columns):
        season_vals = pd.to_numeric(frame["season"], errors="coerce").fillna(0)
        week_vals = pd.to_numeric(frame["week"], errors="coerce").fillna(0)
        order = season_vals * 32 + week_vals
        max_order = float(order.max())
        min_order = float(order.min())
        span = max(max_order - min_order, 1.0)
        age_weeks = (max_order - order) / span
        recency_weight = np.exp(-age_weeks * 0.75)

    if recency_weight.max() > 0:
        recency_weight = recency_weight / recency_weight.max()
    else:
        recency_weight = pd.Series(1.0, index=frame.index, dtype=float)

    usage_weights = {
        "snap_count": 1.2,
        "season_snap_count": 0.6,
        "receiving_targets": 1.5,
        "season_receiving_targets": 0.9,
        "rushing_attempts": 1.0,
        "season_rushing_attempts": 0.6,
        "routes_run": 1.2,
        "season_routes_run": 0.7,
        "touches": 1.2,
        "season_touches": 0.7,
        "fantasy_points": 0.5,
        "season_fantasy_points": 0.3,
    }
    usage_scores = pd.Series(0.0, index=frame.index, dtype=float)
    for col, weight in usage_weights.items():
        if col in frame.columns:
            usage_scores = usage_scores + frame[col].fillna(0).astype(float) * weight

    if usage_scores.max() > 0:
        usage_scores = usage_scores / usage_scores.max()
    else:
        usage_scores = pd.Series(0.0, index=frame.index, dtype=float)

    if {"team", "position"}.issubset(frame.columns):
        team_keys = (
            frame["team"].fillna("").astype(str)
            + "|"
            + frame["position"].fillna("").astype(str)
        )
        group_max = usage_scores.groupby(team_keys).transform(
            lambda s: max(float(s.max()), 1.0)
        )
        usage_share = usage_scores / group_max
    else:
        usage_share = usage_scores

    usage_share = usage_share.clip(lower=0.0, upper=1.0)
    recency_component = recency_weight.clip(lower=1e-3)
    weights = recency_component * (0.35 + 0.65 * usage_share + 0.05)
    if weights.max() > 0:
        weights = weights / weights.max()
    return weights.clip(lower=1e-4)

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
NFL_SEASONS = ["2025-regular", "2024-regular"]

DEFAULT_NFL_API_USER = "4359aa1b-cc29-4647-a3e5-7314e2"
DEFAULT_NFL_API_PASS = "MYSPORTSFEEDS"
DEFAULT_NFL_API_TIMEOUT = 45
DEFAULT_NFL_API_TIMEOUT_RETRIES = 2
DEFAULT_NFL_API_HTTP_RETRIES = 3
DEFAULT_NFL_API_TIMEOUT_BACKOFF = 1.5

ODDS_API_KEY = "5b6f0290e265c3329b3ed27897d79eaf"
ODDS_BASE = "https://api.the-odds-api.com/v4"
NFL_SPORT_KEY = "americanfootball_nfl"
ODDS_GAME_REGIONS = ["us"]
ODDS_PROP_REGIONS = ["us"]
ODDS_EVENT_MARKETS = ("h2h",)
ODDS_BOOKMAKERS = ["draftkings", "fanduel"]
ODDS_FORMAT = "american"
ODDS_DEFAULT_MARKETS = [
    "h2h",
    "totals",
]
# TODO: Expand the TARGET_MARKET_INFO-style metadata (see PLAYER_PROP_MARKET_COLUMN_MAP
# below) once we confirm additional prop coverage from the Odds API. The current list
# intentionally mirrors the minimal set of live markets that we have verified so far.
ODDS_PLAYER_PROP_MARKETS = [
    "player_pass_tds",
    "player_pass_yds",
    "player_rush_tds",
    "player_rush_yds",
    "player_receptions",
    "player_reception_yds",
    "player_anytime_td",
]

# Explicit list of market keys to request from the Odds API. Keeping this separate from
# the canonical list prevents synonyms (e.g. ``*_yards``) from sneaking into outbound
# requests, which previously triggered HTTP 422 errors.
ODDS_PLAYER_PROP_MARKET_REQUEST_KEYS: List[str] = [
    "player_pass_tds",
    "player_pass_yds",
    "player_rush_tds",
    "player_rush_yds",
    "player_receptions",
    "player_reception_yds",
    "player_anytime_td",
]

# Map raw market keys returned by the Odds API to their canonical equivalents.
ODDS_PLAYER_PROP_MARKET_SYNONYMS = {
    # Identity mappings for canonical keys
    **{key: key for key in ODDS_PLAYER_PROP_MARKETS},
    # Legacy / alternate spellings
    "player_pass_yards": "player_pass_yds",
    "player_rush_yards": "player_rush_yds",
    "player_receiving_yards": "player_reception_yds",
    "player_reception_yards": "player_reception_yds",
    "player_rec_yds": "player_reception_yds",
    "player_receiving_yds": "player_reception_yds",
}

# TODO: The downstream ROI evaluation expects every supported market to surface both
# live odds and historical closing lines. Audit this mapping once live prices flow
# consistently so the feature matrix exposes the same set of columns that the odds
# ingestion produces.
PLAYER_PROP_MARKET_COLUMN_MAP = {
    "player_pass_tds": "line_passing_tds",
    "player_pass_yds": "line_passing_yards",
    "player_pass_yards": "line_passing_yards",
    "player_rush_tds": "line_rushing_tds",
    "player_rush_yds": "line_rushing_yards",
    "player_rush_yards": "line_rushing_yards",
    "player_receptions": "line_receptions",
    "player_reception_yds": "line_receiving_yards",
    "player_reception_yards": "line_receiving_yards",
    "player_rec_yds": "line_receiving_yards",
    "player_receiving_yds": "line_receiving_yards",
    "player_receiving_yards": "line_receiving_yards",
    "player_anytime_td": "line_anytime_td",
}

ODDS_MARKET_CANONICAL_MAP = {
    "h2h": "MONEYLINE",
    "spreads": "SPREAD",
    "totals": "TOTAL",
    "player_pass_tds": "PASS_TD",
    "player_pass_yds": "PASS_YDS",
    "player_pass_yards": "PASS_YDS",
    "player_rush_tds": "RUSH_TD",
    "player_rush_yds": "RUSH_YDS",
    "player_rush_yards": "RUSH_YDS",
    "player_receptions": "RECEPTIONS",
    "player_reception_yds": "REC_YDS",
    "player_reception_yards": "REC_YDS",
    "player_rec_yds": "REC_YDS",
    "player_receiving_yds": "REC_YDS",
    "player_receiving_yards": "REC_YDS",
    "player_anytime_td": "ANY_TD",
}

ODDS_PROP_MAX_CONCURRENCY = 1
ODDS_RATE_LIMIT_DELAY = 0.8


def odds_join_param(value: Optional[Union[str, Sequence[str]]]) -> Optional[str]:
    """Convert a sequence or string into a comma-delimited Odds API parameter."""

    if value is None:
        return None
    if isinstance(value, str):
        return value
    joined = ",".join([str(item) for item in value if item])
    return joined or None

odds_logger = logging.getLogger(__name__)


def canonical_prop_market_key(market_key: str) -> str:
    normalized = (market_key or "").lower()
    return ODDS_PLAYER_PROP_MARKET_SYNONYMS.get(normalized, normalized)


def odds_american_to_decimal(american: float) -> float:
    american = float(american)
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def odds_american_to_prob(american: float) -> float:
    return 1.0 / odds_american_to_decimal(american)


def odds_normalize_datetime(value: Optional[dt.datetime]) -> Optional[dt.datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def odds_isoformat(value: Optional[dt.datetime]) -> Optional[str]:
    normalized = odds_normalize_datetime(value)
    if normalized is None:
        return None
    return normalized.isoformat().replace("+00:00", "Z")


def _odds_build_url(path: str, params: Dict[str, Any]) -> str:
    base = ODDS_BASE.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return f"{base}{path}?{urlencode(params, doseq=True)}"


def _merge_odds_event_payload(
    base: Optional[Dict[str, Any]], addition: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Combine two Odds API event payloads, keeping the most recent bookmaker data."""

    if base is None:
        return addition
    if addition is None or base is addition:
        return base

    base.setdefault("bookmakers", [])
    addition_bookmakers = addition.get("bookmakers") or []

    for key in ["id", "sport_key", "commence_time", "home_team", "away_team"]:
        if not base.get(key) and addition.get(key):
            base[key] = addition.get(key)

    bookmaker_map: Dict[str, Dict[str, Any]] = {}
    for bookmaker in base.get("bookmakers", []):
        book_key = bookmaker.get("key") or bookmaker.get("title")
        if not book_key:
            continue
        bookmaker.setdefault("markets", [])
        bookmaker_map[book_key] = bookmaker

    for add_book in addition_bookmakers:
        book_key = add_book.get("key") or add_book.get("title")
        if not book_key:
            continue
        add_book.setdefault("markets", [])
        existing = bookmaker_map.get(book_key)
        if existing is None:
            base["bookmakers"].append(add_book)
            bookmaker_map[book_key] = add_book
            continue

        existing_last = parse_dt(existing.get("last_update")) if existing.get("last_update") else None
        addition_last = parse_dt(add_book.get("last_update")) if add_book.get("last_update") else None
        if addition_last and (existing_last is None or addition_last > existing_last):
            existing["last_update"] = add_book.get("last_update")

        market_map = {
            (market.get("key") or "").lower(): market for market in existing.get("markets", []) or []
        }
        for add_market in add_book.get("markets", []) or []:
            market_key = (add_market.get("key") or "").lower()
            if not market_key:
                continue
            target_market = market_map.get(market_key)
            if target_market is None:
                existing.setdefault("markets", []).append(add_market)
                market_map[market_key] = add_market
                continue

            existing_outcomes = target_market.get("outcomes") or []
            outcome_map = {
                (
                    outcome.get("name"),
                    outcome.get("description"),
                    outcome.get("participant"),
                ): outcome
                for outcome in existing_outcomes
            }
            for add_outcome in add_market.get("outcomes", []) or []:
                key = (
                    add_outcome.get("name"),
                    add_outcome.get("description"),
                    add_outcome.get("participant"),
                )
                if key in outcome_map:
                    outcome_map[key].update({k: v for k, v in add_outcome.items() if v is not None})
                else:
                    existing_outcomes.append(add_outcome)
            target_market["outcomes"] = existing_outcomes

    return base


async def _odds_get_json(
    session: aiohttp.ClientSession,
    path: str,
    api_key: Optional[str],
    *,
    allow_insecure_ssl: bool,
    max_retries: int = 3,
    retry_statuses: Optional[Set[int]] = None,
    **params: Any,
) -> Optional[Any]:
    params = {"apiKey": api_key or ODDS_API_KEY, **params}
    url = _odds_build_url(path, params)
    retries_remaining = max_retries
    retryable = retry_statuses or {429, 500, 502, 503, 504}
    backoff = ODDS_RATE_LIMIT_DELAY

    while True:
        try:
            async with session.get(url, timeout=30) as response:
                if response.status == 200:
                    return await response.json()

                text = await response.text()
                status = response.status
                if status in retryable and retries_remaining > 0:
                    retries_remaining -= 1
                    retry_after = response.headers.get("Retry-After")
                    try:
                        wait_seconds = float(retry_after)
                    except (TypeError, ValueError):
                        wait_seconds = backoff
                    wait_seconds = max(wait_seconds, backoff)
                    odds_logger.warning(
                        "Odds GET %s -> %s (retrying in %.2fs)\n%s",
                        status,
                        url,
                        wait_seconds,
                        text[:600],
                    )
                    await asyncio.sleep(wait_seconds)
                    backoff = min(backoff * 2.0, 5.0)
                    continue

                odds_logger.warning("Odds GET %s -> %s\n%s", status, url, text[:600])
                return None
        except client_exceptions.ClientConnectorCertificateError:
            if not allow_insecure_ssl:
                raise
            odds_logger.exception("GET failed: %s", url)
            return None
        except Exception:
            if retries_remaining > 0:
                retries_remaining -= 1
                odds_logger.warning("Odds request error for %s; retrying", url, exc_info=True)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 5.0)
                continue
            odds_logger.exception("GET failed: %s", url)
            return None


async def odds_fetch_game_odds(
    session: aiohttp.ClientSession,
    *,
    api_key: Optional[str],
    regions: Sequence[str] | str = ("us", "us2"),
    markets: Sequence[str] | str = ("h2h", "spreads", "totals"),
    bookmakers: Sequence[str] | str | None = ODDS_BOOKMAKERS,
    odds_format: str = ODDS_FORMAT,
    date_format: str = "iso",
    allow_insecure_ssl: bool = False,
) -> pd.DataFrame:
    regions_param = odds_join_param(regions)
    markets_param = odds_join_param(markets)
    bookmakers_param = odds_join_param(bookmakers)

    data = await _odds_get_json(
        session,
        f"/sports/{NFL_SPORT_KEY}/odds",
        api_key,
        allow_insecure_ssl=allow_insecure_ssl,
        regions=regions_param,
        markets=markets_param,
        bookmakers=bookmakers_param,
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

    rows: List[Dict[str, Any]] = []
    for event in data:
        event_id = str(event.get("id", ""))
        home_team = event.get("home_team") or event.get("homeTeam") or ""
        away_team = event.get("away_team") or event.get("awayTeam") or ""
        commence_time = event.get("commence_time") or event.get("commenceTime") or ""

        for bookmaker in event.get("bookmakers", []) or []:
            book_key = bookmaker.get("key") or bookmaker.get("title")
            last_update = bookmaker.get("last_update") or bookmaker.get("lastUpdate")
            for market in bookmaker.get("markets", []) or []:
                market_key = str(market.get("key", "")).lower()
                market_label = ODDS_MARKET_CANONICAL_MAP.get(market_key, market_key.upper())
                for outcome in market.get("outcomes", []) or []:
                    outcome_name = outcome.get("name")
                    price = outcome.get("price")
                    line = outcome.get("point") or outcome.get("total") or outcome.get("handicap")

                    rows.append(
                        {
                            "event_id": event_id,
                            "home_team": home_team,
                            "away_team": away_team,
                            "commence_time": commence_time,
                            "book": book_key,
                            "market": market_label,
                            "side": outcome_name,
                            "line": float(line) if line is not None else np.nan,
                            "american_odds": float(price) if price is not None else np.nan,
                            "last_update": last_update,
                        }
                    )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["decimal_odds"] = frame["american_odds"].apply(odds_american_to_decimal)
    frame["imp_prob"] = frame["american_odds"].apply(odds_american_to_prob)
    if "line" in frame.columns:
        frame["line"] = pd.to_numeric(frame["line"], errors="coerce")
    if "last_update" in frame.columns:
        frame["last_update"] = frame["last_update"].astype(str)
    return frame.reset_index(drop=True)


async def _odds_fetch_event_history(
    session: aiohttp.ClientSession,
    event_id: str,
    *,
    api_key: Optional[str],
    regions: Sequence[str] | str,
    markets: Sequence[str] | str,
    bookmakers: Optional[Sequence[str] | str],
    odds_format: str,
    date_format: str,
    allow_insecure_ssl: bool,
) -> Optional[Any]:
    """Fetch odds history for an event and merge snapshots into a single payload."""

    regions_param = odds_join_param(regions)
    markets_param = odds_join_param(markets)
    bookmakers_param = odds_join_param(bookmakers)

    payload = await _odds_get_json(
        session,
        f"/sports/{NFL_SPORT_KEY}/events/{event_id}/odds-history",
        api_key,
        allow_insecure_ssl=allow_insecure_ssl,
        regions=regions_param,
        markets=markets_param,
        bookmakers=bookmakers_param,
        oddsFormat=odds_format,
        dateFormat=date_format,
    )
    await asyncio.sleep(0.3)

    if not payload:
        return None

    # The history endpoint may return either a single snapshot dict or a list of
    # snapshots ordered by update time. Merge everything into a single structure
    # that mirrors the live odds payload shape so downstream parsing logic can
    # remain unchanged.
    if isinstance(payload, dict):
        snapshots = payload.get("data") if "data" in payload else None
        if isinstance(snapshots, list):
            merged: Optional[Dict[str, Any]] = None
            for snap in snapshots:
                if isinstance(snap, dict):
                    merged = _merge_odds_event_payload(merged, snap)
            if merged is not None:
                for key in ["id", "home_team", "away_team", "commence_time", "sport_key"]:
                    if not merged.get(key) and payload.get(key):
                        merged[key] = payload.get(key)
                return merged
        return payload

    if isinstance(payload, list):
        merged = None
        for snap in payload:
            if isinstance(snap, dict):
                merged = _merge_odds_event_payload(merged, snap)
        return merged

    return None


async def odds_fetch_prop_odds(
    session: aiohttp.ClientSession,
    event_ids: Sequence[str] | str,
    *,
    api_key: Optional[str],
    regions: Sequence[str] | str = ("us", "us2"),
    bookmakers: Sequence[str] | str | None = ODDS_BOOKMAKERS,
    odds_format: str = ODDS_FORMAT,
    date_format: str = "iso",
    allow_insecure_ssl: bool = False,
    fallback_to_history: bool = False,
) -> pd.DataFrame:
    if isinstance(event_ids, str):
        event_ids = [event_ids]
    event_ids = [event_id for event_id in (event_ids or []) if event_id]
    if not event_ids:
        return pd.DataFrame(
            columns=[
                "event_id",
                "commence_time",
                "home_team",
                "away_team",
                "book",
                "market",
                "market_display",
                "player",
                "side",
                "line",
                "american_odds",
                "decimal_odds",
                "imp_prob",
            ]
        )

    regions_param = odds_join_param(regions)
    bookmakers_param = odds_join_param(bookmakers)
    request_markets = list(ODDS_PLAYER_PROP_MARKET_REQUEST_KEYS)

    semaphore = asyncio.Semaphore(max(1, ODDS_PROP_MAX_CONCURRENCY))

    async def fetch_event(event_id: str) -> Optional[Any]:
        async with semaphore:
            async def fetch_subset(markets: Sequence[str]) -> Optional[Any]:
                if not markets:
                    return None
                payload = await _odds_get_json(
                    session,
                    f"/sports/{NFL_SPORT_KEY}/events/{event_id}/odds",
                    api_key,
                    allow_insecure_ssl=allow_insecure_ssl,
                    regions=regions_param,
                    markets=",".join(markets),
                    bookmakers=bookmakers_param,
                    oddsFormat=odds_format,
                    dateFormat=date_format,
                )
                await asyncio.sleep(ODDS_RATE_LIMIT_DELAY)
                merged_payload: Optional[Any] = None
                if payload:
                    merged_payload = payload
                elif fallback_to_history:
                    merged_payload = await _odds_fetch_event_history(
                        session,
                        event_id,
                        api_key=api_key,
                        regions=regions,
                        markets=markets,
                        bookmakers=bookmakers,
                        odds_format=odds_format,
                        date_format=date_format,
                        allow_insecure_ssl=allow_insecure_ssl,
                    )
                if merged_payload:
                    return merged_payload
                if len(markets) == 1:
                    odds_logger.debug(
                        "Odds API did not return data for prop market %s on event %s",
                        markets[0],
                        event_id,
                    )
                    return None
                split = max(1, len(markets) // 2)
                left = await fetch_subset(markets[:split])
                right = await fetch_subset(markets[split:])
                return _merge_odds_event_payload(left, right)

            return await fetch_subset(request_markets)

    results = await asyncio.gather(
        *(fetch_event(event_id) for event_id in event_ids), return_exceptions=False
    )

    missing_events = [
        event_id for event_id, payload in zip(event_ids, results) if not payload
    ]
    if missing_events:
        sample = ", ".join(missing_events[:3])
        if len(missing_events) > 3:
            sample = f"{sample}, …"
        odds_logger.info(
            "Live prop odds missing for %d event(s) (sample: %s | markets=%s | bookmakers=%s)",
            len(missing_events),
            sample or "n/a",
            ",".join(request_markets),
            bookmakers_param or "default",
        )

    rows: List[Dict[str, Any]] = []
    for event_id, payload in zip(event_ids, results):
        if not payload:
            continue
        home_team = payload.get("home_team") or payload.get("homeTeam")
        away_team = payload.get("away_team") or payload.get("awayTeam")
        commence_time = payload.get("commence_time") or payload.get("commenceTime")

        for bookmaker in payload.get("bookmakers", []) or []:
            book_key = bookmaker.get("key") or bookmaker.get("title")
            for market in bookmaker.get("markets", []) or []:
                market_key_raw = str(market.get("key", "")).lower()
                canonical_key = canonical_prop_market_key(market_key_raw)
                if canonical_key not in ODDS_PLAYER_PROP_MARKETS:
                    continue
                market_label = ODDS_MARKET_CANONICAL_MAP.get(
                    canonical_key, canonical_key.upper()
                )
                for outcome in market.get("outcomes", []) or []:
                    player = outcome.get("description") or outcome.get("participant") or ""
                    side = (outcome.get("name") or "").title()
                    if canonical_key == "player_anytime_td":
                        side = {"Yes": "Over", "No": "Under"}.get(side, side)
                    price = outcome.get("price")
                    line = outcome.get("point")
                    if not player or price is None:
                        continue
                    rows.append(
                        {
                            "event_id": event_id,
                            "home_team": home_team,
                            "away_team": away_team,
                            "commence_time": commence_time,
                            "book": book_key,
                            "market": canonical_key,
                            "market_display": market_label,
                            "player": str(player),
                            "side": side,
                            "line": float(line) if line is not None else np.nan,
                            "american_odds": float(price),
                            "last_update": bookmaker.get("last_update")
                            or bookmaker.get("lastUpdate"),
                        }
                    )

    frame = pd.DataFrame(rows)
    if frame.empty:
        odds_logger.debug(
            "Odds API returned zero player prop rows for %d event(s) | markets=%s | bookmakers=%s",
            len(event_ids),
            ",".join(request_markets),
            bookmakers_param or "default",
        )
        return frame

    frame["decimal_odds"] = frame["american_odds"].apply(odds_american_to_decimal)
    frame["imp_prob"] = frame["american_odds"].apply(odds_american_to_prob)
    if "line" in frame.columns:
        frame["line"] = pd.to_numeric(frame["line"], errors="coerce")
    if "last_update" in frame.columns:
        frame["last_update"] = frame["last_update"].astype(str)
    return frame.reset_index(drop=True)


async def odds_fetch_event_index(
    session: aiohttp.ClientSession,
    *,
    api_key: Optional[str],
    regions: Sequence[str] | str = ODDS_GAME_REGIONS,
    bookmakers: Sequence[str] | str | None = ODDS_BOOKMAKERS,
    markets: Sequence[str] | str = ODDS_EVENT_MARKETS,
    odds_format: str = ODDS_FORMAT,
    date_format: str = "iso",
    allow_insecure_ssl: bool = False,
) -> pd.DataFrame:
    """Fetch upcoming event metadata (id, teams, commence time)."""

    data = await _odds_get_json(
        session,
        f"/sports/{NFL_SPORT_KEY}/odds",
        api_key,
        allow_insecure_ssl=allow_insecure_ssl,
        regions=odds_join_param(regions),
        markets=odds_join_param(markets),
        bookmakers=odds_join_param(bookmakers),
        oddsFormat=odds_format,
        dateFormat=date_format,
    )

    rows: List[Dict[str, Any]] = []
    for event in data or []:
        rows.append(
            {
                "event_id": str(event.get("id") or ""),
                "home_team": event.get("home_team") or event.get("homeTeam"),
                "away_team": event.get("away_team") or event.get("awayTeam"),
                "commence_time": event.get("commence_time") or event.get("commenceTime"),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame = frame[frame["event_id"].astype(bool)]
    frame = frame.drop_duplicates(subset=["event_id"], keep="last")
    frame["commence_dt"] = pd.to_datetime(frame["commence_time"], errors="coerce", utc=True)
    return frame.reset_index(drop=True)


async def odds_fetch_events_range(
    session: aiohttp.ClientSession,
    *,
    api_key: Optional[str],
    start: Optional[dt.datetime],
    end: Optional[dt.datetime],
    date_format: str = "iso",
    regions: Sequence[str] | str = ODDS_GAME_REGIONS,
    bookmakers: Sequence[str] | str | None = ODDS_BOOKMAKERS,
    markets: Sequence[str] | str = ODDS_EVENT_MARKETS,
    odds_format: str = ODDS_FORMAT,
    allow_insecure_ssl: bool = False,
) -> pd.DataFrame:
    start_dt = odds_normalize_datetime(start) or (default_now_utc() - dt.timedelta(days=7))
    end_dt = odds_normalize_datetime(end) or (start_dt + dt.timedelta(days=14))
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt

    rows: List[Dict[str, Any]] = []
    current_date = start_dt.date()
    end_date = end_dt.date()

    while current_date <= end_date:
        day_start = dt.datetime.combine(current_date, dt.time.min, tzinfo=dt.timezone.utc)
        day_end = day_start + dt.timedelta(days=1)
        payload = await _odds_get_json(
            session,
            f"/sports/{NFL_SPORT_KEY}/odds",
            api_key,
            allow_insecure_ssl=allow_insecure_ssl,
            dateFormat=date_format,
            commenceTimeFrom=odds_isoformat(day_start),
            commenceTimeTo=odds_isoformat(day_end),
            regions=odds_join_param(regions),
            bookmakers=odds_join_param(bookmakers),
            markets=odds_join_param(markets),
            oddsFormat=odds_format,
        )
        if not payload:
            payload = await _odds_get_json(
                session,
                f"/sports/{NFL_SPORT_KEY}/events",
                api_key,
                allow_insecure_ssl=allow_insecure_ssl,
                dateFormat=date_format,
                commenceTimeFrom=odds_isoformat(day_start),
                commenceTimeTo=odds_isoformat(day_end),
            )
        if payload:
            for event in payload:
                rows.append(
                    {
                        "event_id": str(event.get("id") or ""),
                        "commence_time": event.get("commence_time")
                        or event.get("commenceTime"),
                        "home_team": event.get("home_team") or event.get("homeTeam"),
                        "away_team": event.get("away_team") or event.get("awayTeam"),
                    }
                )
        await asyncio.sleep(0.25)
        current_date += dt.timedelta(days=1)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["event_id"].astype(bool)]
    df = df.drop_duplicates(subset=["event_id"], keep="last")
    df["commence_dt"] = pd.to_datetime(df["commence_time"], errors="coerce", utc=True)
    return df.reset_index(drop=True)


async def odds_fetch_event_game_markets(
    session: aiohttp.ClientSession,
    event_ids: Sequence[str] | str,
    *,
    api_key: Optional[str],
    regions: Sequence[str] | str = ("us",),
    markets: Sequence[str] | str = ("h2h", "totals"),
    bookmakers: Sequence[str] | str | None = ODDS_BOOKMAKERS,
    odds_format: str = ODDS_FORMAT,
    date_format: str = "iso",
    allow_insecure_ssl: bool = False,
    fallback_to_history: bool = False,
) -> pd.DataFrame:
    if isinstance(event_ids, str):
        event_ids = [event_ids]
    event_ids = [event_id for event_id in (event_ids or []) if event_id]
    if not event_ids:
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
                "last_update",
            ]
        )

    regions_param = odds_join_param(regions)
    markets_param = odds_join_param(markets)
    bookmakers_param = odds_join_param(bookmakers)
    markets_filter = {
        m.lower() for m in (markets if isinstance(markets, (list, tuple)) else [markets])
    }

    semaphore = asyncio.Semaphore(2)

    async def fetch_event(event_id: str) -> Optional[Any]:
        async with semaphore:
            payload = await _odds_get_json(
                session,
                f"/sports/{NFL_SPORT_KEY}/events/{event_id}/odds",
                api_key,
                allow_insecure_ssl=allow_insecure_ssl,
                regions=regions_param,
                markets=markets_param,
                bookmakers=bookmakers_param,
                oddsFormat=odds_format,
                dateFormat=date_format,
            )
            await asyncio.sleep(ODDS_RATE_LIMIT_DELAY)
            if payload:
                return payload
            if not fallback_to_history:
                return None
            return await _odds_fetch_event_history(
                session,
                event_id,
                api_key=api_key,
                regions=regions,
                markets=markets,
                bookmakers=bookmakers,
                odds_format=odds_format,
                date_format=date_format,
                allow_insecure_ssl=allow_insecure_ssl,
            )

    results = await asyncio.gather(
        *(fetch_event(event_id) for event_id in event_ids), return_exceptions=False
    )

    rows: List[Dict[str, Any]] = []
    for event_id, payload in zip(event_ids, results):
        if not payload:
            continue
        home_team = payload.get("home_team") or payload.get("homeTeam")
        away_team = payload.get("away_team") or payload.get("awayTeam")
        commence_time = payload.get("commence_time") or payload.get("commenceTime")

        for bookmaker in payload.get("bookmakers", []) or []:
            book_key = bookmaker.get("key") or bookmaker.get("title")
            last_update = bookmaker.get("last_update") or bookmaker.get("lastUpdate")
            for market in bookmaker.get("markets", []) or []:
                market_key = str(market.get("key", "")).lower()
                if market_key not in markets_filter:
                    continue
                market_label = ODDS_MARKET_CANONICAL_MAP.get(market_key, market_key.upper())
                for outcome in market.get("outcomes", []) or []:
                    outcome_name = outcome.get("name")
                    price = outcome.get("price")
                    line = outcome.get("point") or outcome.get("total") or outcome.get("handicap")
                    rows.append(
                        {
                            "event_id": event_id,
                            "home_team": home_team,
                            "away_team": away_team,
                            "commence_time": commence_time,
                            "book": book_key,
                            "market": market_label,
                            "side": outcome_name,
                            "line": float(line) if line is not None else np.nan,
                            "american_odds": float(price) if price is not None else np.nan,
                            "last_update": last_update,
                        }
                    )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["decimal_odds"] = df["american_odds"].apply(odds_american_to_decimal)
    df["imp_prob"] = df["american_odds"].apply(odds_american_to_prob)
    if "line" in df.columns:
        df["line"] = pd.to_numeric(df["line"], errors="coerce")
    df["commence_dt"] = pd.to_datetime(df["commence_time"], errors="coerce", utc=True)
    if "last_update" in df.columns:
        df["last_update"] = df["last_update"].astype(str)
    return df.reset_index(drop=True)

def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


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

CITY_NAME_TO_ABBR = {
    "arizona": "ARI",
    "atlanta": "ATL",
    "baltimore": "BAL",
    "buffalo": "BUF",
    "carolina": "CAR",
    "chicago": "CHI",
    "cincinnati": "CIN",
    "cleveland": "CLE",
    "dallas": "DAL",
    "denver": "DEN",
    "detroit": "DET",
    "green bay": "GB",
    "houston": "HOU",
    "indianapolis": "IND",
    "jacksonville": "JAX",
    "kansas city": "KC",
    "las vegas": "LV",
    "miami": "MIA",
    "minnesota": "MIN",
    "new england": "NE",
    "new orleans": "NO",
    "philadelphia": "PHI",
    "pittsburgh": "PIT",
    "san francisco": "SF",
    "seattle": "SEA",
    "tampa bay": "TB",
    "tennessee": "TEN",
    "washington": "WAS",
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

    if sanitized in CITY_NAME_TO_ABBR:
        return CITY_NAME_TO_ABBR[sanitized]

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

    tokens = sanitized.split()
    for token in tokens:
        if token in TEAM_MASCOT_TO_ABBR:
            return TEAM_MASCOT_TO_ABBR[token]

    # As a last resort, try to map by taking the first letter of each token
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

NON_NEGATIVE_TARGETS: set[str] = set(TARGET_ALLOWED_POSITIONS.keys())


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

    # First try an exact alias match (e.g., HB -> RB).
    text = POSITION_ALIAS_MAP.get(text, text)

    # Many feeds (including MSF lineups) encode positions with prefixes such as
    # "Offense-WR-1" or "SpecialTeams-K-1".  Break the string into alpha-only
    # tokens so we can match the meaningful portion (WR, QB, etc.).
    tokens = [text]
    tokens.extend(token for token in re.split(r"[^A-Z]", text) if token)

    for token in tokens:
        # Allow alias substitution on each token (e.g., SLOT -> WR).
        if token in POSITION_ALIAS_MAP:
            return POSITION_ALIAS_MAP[token]
        for prefix, canonical in POSITION_PREFIX_MAP.items():
            if token.startswith(prefix):
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
    working: pd.DataFrame, lineup_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    No-op: we no longer synthesize placeholder rows from lineup entries.
    We only keep players who already exist with real production rows.
    """

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
        self.closing_odds_records = self._load_records(config.closing_odds_history_path)
        self.travel_context_records = self._load_records(config.rest_travel_context_path)

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

        self.closing_odds_frame = self._build_closing_odds_frame(self.closing_odds_records)
        self.travel_context_frame = self._build_travel_context_frame(self.travel_context_records)

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

    def _build_closing_odds_frame(
        self, records: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        columns = [
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "home_closing_moneyline",
            "away_closing_moneyline",
            "home_closing_implied_prob",
            "away_closing_implied_prob",
            "closing_bookmaker",
            "closing_line_time",
        ]

        if not records:
            return pd.DataFrame(columns=columns)

        frame = pd.DataFrame(records)
        normalized = _standardize_closing_odds_frame(frame, "Local CSV")
        if normalized.empty:
            return pd.DataFrame(columns=columns)

        result = normalized.copy()
        result["season"] = result["season"].astype(str).str.strip()
        result["home_team"] = result["home_team"].apply(normalize_team_abbr)
        result["away_team"] = result["away_team"].apply(normalize_team_abbr)
        if "week" in result.columns:
            result["week"] = result["week"].apply(
                lambda x: int(x) if pd.notna(x) else None
            )
        else:
            result["week"] = None

        for side in ("home", "away"):
            ml_col = f"{side}_closing_moneyline"
            prob_col = f"{side}_closing_implied_prob"
            result[ml_col] = pd.to_numeric(result.get(ml_col), errors="coerce")
            result[prob_col] = result[ml_col].apply(
                lambda val: odds_american_to_prob(float(val)) if pd.notna(val) else np.nan
            )

        if "closing_bookmaker" in result.columns:
            result["closing_bookmaker"] = result["closing_bookmaker"].fillna("Local CSV")
        else:
            result["closing_bookmaker"] = "Local CSV"

        if "closing_line_time" in result.columns:
            result["closing_line_time"] = pd.to_datetime(
                result["closing_line_time"], errors="coerce", utc=True
            )
        else:
            result["closing_line_time"] = pd.NaT

        if "game_id" not in result.columns:
            result["game_id"] = None

        # Drop rows that still lack a team mapping or both moneylines after normalization;
        # these cannot be merged back to the schedule and only generate duplicates.
        required_team_cols = ["home_team", "away_team"]
        required_odds_cols = ["home_closing_moneyline", "away_closing_moneyline"]
        drop_mask = result[required_team_cols].isna().any(axis=1)
        drop_mask |= result[required_odds_cols].isna().all(axis=1)
        if drop_mask.any():
            result = result.loc[~drop_mask]

        missing_cols = [col for col in columns if col not in result.columns]
        for col in missing_cols:
            result[col] = pd.NaT if col.endswith("time") else None

        return result[columns]

    def _build_travel_context_frame(
        self, records: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        if not records:
            return pd.DataFrame(
                columns=[
                    "season",
                    "week",
                    "team",
                    "opponent",
                    "rest_days",
                    "rest_penalty",
                    "travel_penalty",
                    "timezone_diff_hours",
                    "avg_timezone_diff_hours",
                ]
            )

        frame = pd.DataFrame(records)
        frame["team"] = frame["team"].apply(normalize_team_abbr)
        if "opponent" in frame.columns:
            frame["opponent"] = frame["opponent"].apply(normalize_team_abbr)
        frame["season"] = frame["season"].astype(str)
        if "week" in frame.columns:
            frame["week"] = frame["week"].apply(lambda x: int(x) if pd.notna(x) else None)
        numeric_cols = [
            "rest_days",
            "rest_penalty",
            "travel_penalty",
            "timezone_diff_hours",
            "avg_timezone_diff_hours",
        ]
        for col in numeric_cols:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame

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
        entry = {
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
        results.append(entry)
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
    closing_odds_history_path: Optional[str] = os.getenv("NFL_CLOSING_ODDS_PATH") or DEFAULT_CLOSING_ODDS_PATH
    rest_travel_context_path: Optional[str] = os.getenv("NFL_TRAVEL_CONTEXT_PATH") or DEFAULT_TRAVEL_CONTEXT_PATH
    msf_user: str = os.getenv("NFL_API_USER", DEFAULT_NFL_API_USER)
    msf_password: str = os.getenv("NFL_API_PASS", DEFAULT_NFL_API_PASS)
    msf_timeout_seconds: int = int(os.getenv("NFL_API_TIMEOUT", str(DEFAULT_NFL_API_TIMEOUT)))
    msf_timeout_retries: int = int(os.getenv("NFL_API_TIMEOUT_RETRIES", str(DEFAULT_NFL_API_TIMEOUT_RETRIES)))
    msf_timeout_backoff: float = float(
        os.getenv("NFL_API_TIMEOUT_BACKOFF", str(DEFAULT_NFL_API_TIMEOUT_BACKOFF))
    )
    msf_http_retries: int = int(os.getenv("NFL_API_HTTP_RETRIES", str(DEFAULT_NFL_API_HTTP_RETRIES)))
    respect_lineups: bool = True
    odds_allow_insecure_ssl: bool = env_flag("ODDS_ALLOW_INSECURE_SSL", False)
    odds_ssl_cert_path: Optional[str] = os.getenv("NFL_ODDS_SSL_CERT")
    enable_paper_trading: bool = env_flag("NFL_PAPER_TRADE", False)
    paper_trade_lookback_days: int = 21
    paper_trade_edge_threshold: float = 0.02
    paper_trade_bankroll: float = 1_000.0
    paper_trade_max_fraction: float = 0.05
    closing_odds_provider: Optional[str] = os.getenv("NFL_CLOSING_ODDS_PROVIDER") or "local"
    closing_odds_timeout: int = int(os.getenv("NFL_CLOSING_ODDS_TIMEOUT", "45"))
    closing_odds_download_dir: Optional[str] = os.getenv("NFL_CLOSING_ODDS_DOWNLOAD_DIR")
    oddsportal_base_url: str = os.getenv(
        "ODDSPORTAL_BASE_URL",
        "https://www.oddsportal.com/american-football/usa/",
    )
    oddsportal_results_path: str = os.getenv(
        "ODDSPORTAL_RESULTS_PATH",
        "nfl/results/",
    )
    oddsportal_season_template: str = os.getenv(
        "ODDSPORTAL_SEASON_TEMPLATE",
        "nfl-{season}/results/",
    )
    oddsportal_user_agents: Tuple[str, ...] = tuple(
        ua.strip()
        for ua in re.split(
            r"[;,]",
            os.getenv("ODDSPORTAL_USER_AGENTS", ""),
        )
        if ua.strip()
    )
    killersports_base_url: Optional[str] = os.getenv("KILLERSPORTS_BASE_URL")
    killersports_api_key: Optional[str] = os.getenv("KILLERSPORTS_API_KEY")
    killersports_username: Optional[str] = os.getenv("KILLERSPORTS_USERNAME")
    killersports_password: Optional[str] = os.getenv("KILLERSPORTS_PASSWORD")

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
        if "odds_event_id" not in game_columns:
            statements.append("ALTER TABLE nfl_games ADD COLUMN IF NOT EXISTS odds_event_id TEXT")
        if "home_closing_moneyline" not in game_columns:
            statements.append(
                "ALTER TABLE nfl_games ADD COLUMN IF NOT EXISTS home_closing_moneyline DOUBLE PRECISION"
            )
        if "away_closing_moneyline" not in game_columns:
            statements.append(
                "ALTER TABLE nfl_games ADD COLUMN IF NOT EXISTS away_closing_moneyline DOUBLE PRECISION"
            )
        if "home_closing_implied_prob" not in game_columns:
            statements.append(
                "ALTER TABLE nfl_games ADD COLUMN IF NOT EXISTS home_closing_implied_prob DOUBLE PRECISION"
            )
        if "away_closing_implied_prob" not in game_columns:
            statements.append(
                "ALTER TABLE nfl_games ADD COLUMN IF NOT EXISTS away_closing_implied_prob DOUBLE PRECISION"
            )
        if "closing_bookmaker" not in game_columns:
            statements.append(
                "ALTER TABLE nfl_games ADD COLUMN IF NOT EXISTS closing_bookmaker TEXT"
            )
        if "closing_line_time" not in game_columns:
            statements.append(
                "ALTER TABLE nfl_games ADD COLUMN IF NOT EXISTS closing_line_time TIMESTAMPTZ"
            )

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

        if "nfl_team_advanced_metrics" in table_names:
            try:
                advanced_columns = {
                    col["name"] for col in inspector.get_columns("nfl_team_advanced_metrics")
                }
            except Exception:
                advanced_columns = set()
            advanced_column_defs = {
                "offense_yards_per_play": "DOUBLE PRECISION",
                "defense_yards_per_play": "DOUBLE PRECISION",
                "offense_td_rate": "DOUBLE PRECISION",
                "defense_td_rate": "DOUBLE PRECISION",
                "pass_rate": "DOUBLE PRECISION",
                "rush_rate": "DOUBLE PRECISION",
                "pass_rate_over_expectation": "DOUBLE PRECISION",
            }
            for column_name, column_type in advanced_column_defs.items():
                if column_name not in advanced_columns:
                    statements.append(
                        f"ALTER TABLE nfl_team_advanced_metrics ADD COLUMN IF NOT EXISTS {column_name} {column_type}"
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
            Column("odds_event_id", String),
            Column("home_moneyline", Float),
            Column("away_moneyline", Float),
            Column("home_implied_prob", Float),
            Column("away_implied_prob", Float),
            Column("home_closing_moneyline", Float),
            Column("away_closing_moneyline", Float),
            Column("home_closing_implied_prob", Float),
            Column("away_closing_implied_prob", Float),
            Column("closing_bookmaker", String),
            Column("closing_line_time", DateTime(timezone=True)),
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
            Column("offense_yards_per_play", Float),
            Column("defense_yards_per_play", Float),
            Column("offense_td_rate", Float),
            Column("defense_td_rate", Float),
            Column("pass_rate", Float),
            Column("rush_rate", Float),
            Column("pass_rate_over_expectation", Float),
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

        self.game_totals = Table(
            "nfl_game_totals",
            self.meta,
            Column("market_id", String, primary_key=True),
            Column("game_id", String, nullable=False),
            Column("event_id", String),
            Column("season", String),
            Column("week", Integer),
            Column("start_time", DateTime(timezone=True)),
            Column("bookmaker", String),
            Column("total_line", Float),
            Column("over_odds", Float),
            Column("under_odds", Float),
            Column("last_update", DateTime(timezone=True)),
            Column("ingested_at", DateTime(timezone=True), default=default_now_utc),
        )

        self.player_prop_lines = Table(
            "nfl_player_prop_lines",
            self.meta,
            Column("prop_id", String, primary_key=True),
            Column("game_id", String, nullable=False),
            Column("event_id", String),
            Column("player_id", String),
            Column("player_name", String),
            Column("player_name_norm", String),
            Column("team", String),
            Column("opponent", String),
            Column("market", String, nullable=False),
            Column("line", Float),
            Column("over_odds", Float),
            Column("under_odds", Float),
            Column("bookmaker", String),
            Column("last_update", DateTime(timezone=True)),
            Column("ingested_at", DateTime(timezone=True), default=default_now_utc),
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

    def fetch_existing_game_ids(self) -> set[str]:
        with self.engine.begin() as conn:
            rows = conn.execute(select(self.games.c.game_id)).fetchall()
        return {row[0] for row in rows}

    def fetch_game_lookup(self) -> List[Dict[str, Any]]:
        columns = [
            self.games.c.game_id,
            self.games.c.season,
            self.games.c.week,
            self.games.c.start_time,
            self.games.c.home_team,
            self.games.c.away_team,
            self.games.c.odds_event_id,
            self.games.c.home_moneyline,
            self.games.c.away_moneyline,
            self.games.c.home_implied_prob,
            self.games.c.away_implied_prob,
            self.games.c.home_closing_moneyline,
            self.games.c.away_closing_moneyline,
            self.games.c.home_closing_implied_prob,
            self.games.c.away_closing_implied_prob,
            self.games.c.closing_bookmaker,
            self.games.c.closing_line_time,
            self.games.c.odds_updated,
        ]
        with self.engine.begin() as conn:
            rows = conn.execute(select(*columns)).fetchall()
        results: List[Dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "game_id": row[0],
                    "season": row[1],
                    "week": row[2],
                    "start_time": row[3],
                    "home_team": row[4],
                    "away_team": row[5],
                    "odds_event_id": row[6],
                    "home_moneyline": row[7],
                    "away_moneyline": row[8],
                    "home_implied_prob": row[9],
                    "away_implied_prob": row[10],
                    "home_closing_moneyline": row[11],
                    "away_closing_moneyline": row[12],
                    "home_closing_implied_prob": row[13],
                    "away_closing_implied_prob": row[14],
                    "closing_bookmaker": row[15],
                    "closing_line_time": row[16],
                    "odds_updated": row[17],
                }
            )
        return results

    def fetch_games_with_odds_for_seasons(self, seasons: Iterable[str]) -> pd.DataFrame:
        season_list = [str(season) for season in seasons if season is not None]
        if not season_list:
            return pd.DataFrame(
                columns=[
                    "game_id",
                    "season",
                    "week",
                    "start_time",
                    "home_team",
                    "away_team",
                    "home_moneyline",
                    "away_moneyline",
                    "odds_updated",
                    "odds_event_id",
                ]
            )

        columns = [
            self.games.c.game_id,
            self.games.c.season,
            self.games.c.week,
            self.games.c.start_time,
            self.games.c.home_team,
            self.games.c.away_team,
            self.games.c.home_moneyline,
            self.games.c.away_moneyline,
            self.games.c.odds_updated,
            self.games.c.odds_event_id,
        ]

        with self.engine.begin() as conn:
            stmt = select(*columns).where(self.games.c.season.in_(season_list))
            rows = conn.execute(stmt).fetchall()

        frame = pd.DataFrame(
            rows,
            columns=[
                "game_id",
                "season",
                "week",
                "start_time",
                "home_team",
                "away_team",
                "home_moneyline",
                "away_moneyline",
                "odds_updated",
                "odds_event_id",
            ],
        )
        if not frame.empty:
            frame["start_time"] = pd.to_datetime(frame["start_time"], errors="coerce")
        return frame

    def fetch_games_with_player_stats(self) -> set[str]:
        """Return the set of game IDs that already have player statistics stored."""

        with self.engine.begin() as conn:
            rows = conn.execute(select(self.player_stats.c.game_id).distinct()).fetchall()
        return {row[0] for row in rows}

    def fetch_existing_advanced_metric_keys(self) -> Set[Tuple[str, int, str]]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(
                    self.team_advanced_metrics.c.season,
                    self.team_advanced_metrics.c.week,
                    self.team_advanced_metrics.c.team,
                )
            ).fetchall()
        keys: Set[Tuple[str, int, str]] = set()
        for season, week, team in rows:
            if team is None or week is None:
                continue
            keys.add((str(season), int(week), normalize_team_abbr(team)))
        return keys

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
        rows: List[Dict[str, Any]] = []

        def _append_metric(metric_name: str, metric_value: Any) -> None:
            if metric_value is None:
                return
            try:
                value_float = float(metric_value)
            except (TypeError, ValueError):
                return
            if math.isnan(value_float):
                return
            rows.append(
                {
                    "run_id": run_id,
                    "model_name": model_name,
                    "metric_name": metric_name,
                    "metric_value": value_float,
                    "sample_size": sample_size,
                }
            )

        for metric, value in metrics.items():
            if isinstance(value, (tuple, list)):
                if len(value) == 2:
                    _append_metric(f"{metric}_lower", value[0])
                    _append_metric(f"{metric}_upper", value[1])
                else:
                    for idx, component in enumerate(value):
                        _append_metric(f"{metric}_{idx}", component)
            else:
                _append_metric(metric, value)

        if not rows:
            return
        with self.engine.begin() as conn:
            conn.execute(self.model_backtests.insert(), rows)


# ---------------------------------------------------------------------------
# API clients
# ---------------------------------------------------------------------------


class MySportsFeedsClient:
    def __init__(
        self,
        user: str,
        password: str,
        *,
        timeout: int = 30,
        timeout_retries: int = 0,
        timeout_backoff: float = 1.0,
        http_retries: int = 3,
    ):
        self.user = user
        self.password = password
        self.auth = (user, password)
        self.timeout = timeout
        self.timeout_retries = max(0, timeout_retries)
        self.timeout_backoff = max(0.0, timeout_backoff)
        self.session = requests.Session()
        retry = Retry(
            total=max(0, http_retries),
            read=max(0, http_retries),
            connect=max(0, http_retries),
            status=max(0, http_retries),
            allowed_methods=("GET",),
            status_forcelist=(429, 500, 502, 503, 504),
            backoff_factor=self.timeout_backoff,
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _request(self, endpoint: str, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{API_PREFIX_NFL}/{endpoint}"
        logging.debug("Requesting MySportsFeeds endpoint %s", url)
        response = None
        for attempt in range(self.timeout_retries + 1):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    auth=self.auth,
                    timeout=self.timeout,
                )
                break
            except ReadTimeout:
                if attempt >= self.timeout_retries:
                    logging.warning(
                        "MySportsFeeds request to %s timed out after %ss; returning empty payload",
                        url,
                        self.timeout,
                    )
                    return {}
                sleep_for = self.timeout_backoff * (2 ** attempt)
                logging.warning(
                    "MySportsFeeds request to %s timed out after %ss (attempt %d/%d); retrying in %.1fs",
                    url,
                    self.timeout,
                    attempt + 1,
                    self.timeout_retries + 1,
                    sleep_for,
                )
                time.sleep(sleep_for)
            except RequestsConnectionError as exc:
                logging.warning("MySportsFeeds request to %s failed: %s", url, exc)
                return {}
        if response is None:
            return {}
        try:
            response.raise_for_status()
        except HTTPError:
            if response.status_code == HTTPStatus.UNAUTHORIZED:
                logging.error(
                    "MySportsFeeds rejected the provided credentials. Set NFL_API_USER/NFL_API_PASS or update config."
                )
            raise
        try:
            return response.json()
        except RequestsJSONDecodeError:
            content = response.text.strip()
            if not content:
                logging.debug(
                    "Empty response body for MySportsFeeds endpoint %s; returning empty payload",
                    url,
                )
                return {}
            logging.warning(
                "Failed to decode JSON from MySportsFeeds endpoint %s (content-type=%s)",
                url,
                response.headers.get("Content-Type"),
            )
            raise

    def fetch_games(self, season: str) -> List[Dict[str, Any]]:
        """Fetch the schedule for a season, retrying with alternative filters."""

        base_params: Dict[str, Any] = {"limit": 500}
        attempts: Tuple[Optional[str], ...] = (
            "completed,upcoming",
            "final,inprogress,scheduled",
            "scheduled",
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

    def fetch_games_by_date_range(
        self,
        season: str,
        start_date: dt.date,
        end_date: dt.date,
        status: Optional[str] = "scheduled",
    ) -> List[Dict[str, Any]]:
        """Fetch games for each date in the provided range (inclusive)."""

        aggregated: List[Dict[str, Any]] = []
        seen_ids: Set[Any] = set()
        current = start_date

        while current <= end_date:
            params: Dict[str, Any] = {
                "limit": 500,
                "date": current.strftime("%Y%m%d"),
            }
            if status:
                params["status"] = status

            try:
                data = self._request(f"{season}/games.json", params=params)
            except Exception:
                logging.debug(
                    "MySportsFeeds date-range request failed for %s on %s",
                    season,
                    current,
                    exc_info=True,
                )
                current += dt.timedelta(days=1)
                continue

            games = data.get("games", [])
            for game in games:
                schedule = game.get("schedule") or {}
                game_id = schedule.get("id")
                if game_id in seen_ids:
                    continue
                aggregated.append(game)
                seen_ids.add(game_id)

            current += dt.timedelta(days=1)

        return aggregated

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
    def __init__(self, api_key: str, timeout: int = 30, allow_insecure_ssl: bool = False):
        self.api_key = api_key
        self.timeout = timeout
        self.allow_insecure_ssl = allow_insecure_ssl

    def _build_connector(self, allow_insecure: bool) -> Optional[aiohttp.TCPConnector]:
        if allow_insecure:
            return aiohttp.TCPConnector(ssl=False)
        try:
            ssl_context = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            ssl_context = None
        if ssl_context is None:
            return None
        return aiohttp.TCPConnector(ssl=ssl_context)

    @staticmethod
    def _normalize_bound(value: Optional[dt.datetime]) -> Optional[dt.datetime]:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=dt.timezone.utc)
        return value.astimezone(dt.timezone.utc)

    @staticmethod
    def _to_iso(value: Optional[dt.datetime]) -> Optional[str]:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=dt.timezone.utc)
        return value.astimezone(dt.timezone.utc).isoformat().replace('+00:00', 'Z')

    @staticmethod
    def _reverse_market_map() -> Dict[str, str]:
        reverse: Dict[str, str] = {}
        for raw_key, canonical in ODDS_MARKET_CANONICAL_MAP.items():
            preferred = raw_key
            if preferred.endswith('_yds'):
                preferred = preferred.replace('_yds', '_yards')
            if preferred == 'player_reception_yds':
                preferred = 'player_receiving_yards'
            reverse[canonical.upper()] = preferred
        return reverse

    def _assemble_events(
        self,
        meta_df: Optional[pd.DataFrame],
        game_df: pd.DataFrame,
        prop_df: pd.DataFrame,
        start: Optional[dt.datetime],
        end: Optional[dt.datetime],
    ) -> List[Dict[str, Any]]:
        reverse_map = self._reverse_market_map()

        def normalize_market(label: str) -> str:
            key = reverse_map.get(label.upper(), label.lower())
            if key.endswith('_yds'):
                key = key.replace('_yds', '_yards')
            if key == 'player_reception_yds':
                key = 'player_receiving_yards'
            return key

        event_meta: Dict[str, Dict[str, Any]] = {}

        # TODO: Guard against index alignment issues here. The Odds API can drop games
        # between calls, so any upstream filtering that mutates ``frame.index`` risks
        # leaking stale labels into ``event_meta`` if we ever rely on positional joins.
        def update_meta(frame: pd.DataFrame) -> None:
            if frame is None or frame.empty:
                return
            for _, row in frame.iterrows():
                eid = str(row.get('event_id') or '')
                if not eid:
                    continue
                meta = event_meta.setdefault(eid, {})
                if not meta.get('home_team') and row.get('home_team'):
                    meta['home_team'] = row.get('home_team')
                if not meta.get('away_team') and row.get('away_team'):
                    meta['away_team'] = row.get('away_team')
                commence_dt = row.get('commence_dt')
                if pd.notna(commence_dt) and meta.get('commence_dt') is None:
                    if isinstance(commence_dt, pd.Timestamp):
                        meta['commence_dt'] = commence_dt.to_pydatetime()
                    elif isinstance(commence_dt, dt.datetime):
                        meta['commence_dt'] = commence_dt
                if not meta.get('commence_raw') and row.get('commence_time'):
                    meta['commence_raw'] = row.get('commence_time')

        update_meta(meta_df)
        update_meta(game_df)
        update_meta(prop_df)

        events: Dict[str, Dict[str, Any]] = {}
        bookmaker_maps: Dict[str, Dict[str, Dict[str, Any]]] = {}

        def get_event(event_id: str) -> Dict[str, Any]:
            if event_id not in events:
                meta = event_meta.get(event_id, {})
                commence_dt = meta.get('commence_dt')
                events[event_id] = {
                    'id': event_id,
                    'commence_time': self._to_iso(commence_dt) if commence_dt else meta.get('commence_raw'),
                    'home_team': meta.get('home_team'),
                    'away_team': meta.get('away_team'),
                    'bookmakers': [],
                }
                bookmaker_maps[event_id] = {}
            return events[event_id]

        def get_bookmaker(event_id: str, book: str) -> Dict[str, Any]:
            event = get_event(event_id)
            book_map = bookmaker_maps[event_id]
            if book not in book_map:
                bookmaker = {
                    'key': book,
                    'title': book,
                    'last_update': self._to_iso(default_now_utc()),
                    'markets': [],
                }
                bookmaker['__markets'] = {}
                book_map[book] = bookmaker
                event['bookmakers'].append(bookmaker)
            return book_map[book]

        def add_outcome(book: Dict[str, Any], market_key: str, outcome: Dict[str, Any]) -> None:
            markets_map = book.setdefault('__markets', {})
            market_struct = markets_map.get(market_key)
            if market_struct is None:
                market_struct = {'key': market_key, 'outcomes': []}
                markets_map[market_key] = market_struct
                book['markets'].append(market_struct)
            market_struct['outcomes'].append(outcome)

        def assign_last_update(book: Dict[str, Any], raw_value: Any) -> None:
            if raw_value is None:
                return
            if isinstance(raw_value, float) and math.isnan(raw_value):
                return
            if isinstance(raw_value, pd.Timestamp):
                iso_val = self._to_iso(raw_value.to_pydatetime())
            elif isinstance(raw_value, dt.datetime):
                iso_val = self._to_iso(raw_value)
            else:
                text = str(raw_value).strip()
                if not text or text.lower() == 'nan':
                    return
                iso_val = text
            if iso_val:
                book['last_update'] = iso_val

        if game_df is not None and not game_df.empty:
            for _, row in game_df.iterrows():
                eid = str(row.get('event_id') or '')
                book_name = str(row.get('book') or '')
                if not eid or not book_name:
                    continue
                price = row.get('american_odds')
                if pd.isna(price):
                    continue
                book = get_bookmaker(eid, book_name)
                assign_last_update(book, row.get('last_update'))
                market_key = normalize_market(str(row.get('market') or ''))
                outcome: Dict[str, Any] = {
                    'name': row.get('side'),
                    'price': float(price),
                }
                line_val = row.get('line')
                if not pd.isna(line_val):
                    outcome['point'] = float(line_val)
                    if market_key == 'totals':
                        outcome['total'] = float(line_val)
                add_outcome(book, market_key, outcome)

        if prop_df is not None and not prop_df.empty:
            for _, row in prop_df.iterrows():
                eid = str(row.get('event_id') or '')
                book_name = str(row.get('book') or '')
                if not eid or not book_name:
                    continue
                price = row.get('american_odds')
                if pd.isna(price):
                    continue
                book = get_bookmaker(eid, book_name)
                assign_last_update(book, row.get('last_update'))
                market_key = normalize_market(str(row.get('market') or ''))
                outcome: Dict[str, Any] = {
                    'name': row.get('side'),
                    'price': float(price),
                    'description': row.get('player'),
                    'participant': row.get('player'),
                }
                line_val = row.get('line')
                if not pd.isna(line_val):
                    outcome['point'] = float(line_val)
                add_outcome(book, market_key, outcome)

        assembled = list(events.values())
        for event in assembled:
            for bookmaker in event.get('bookmakers', []):
                bookmaker.pop('__markets', None)

        if start or end:
            filtered: List[Dict[str, Any]] = []
            start_utc = self._normalize_bound(start)
            end_utc = self._normalize_bound(end)
            for event in assembled:
                commence = parse_dt(event.get('commence_time'))
                if start_utc and (commence is None or commence < start_utc):
                    continue
                if end_utc and (commence is None or commence > end_utc):
                    continue
                filtered.append(event)
            assembled = filtered
        return assembled

    async def _fetch_async(
        self,
        start: Optional[dt.datetime],
        end: Optional[dt.datetime],
        include_player_props: bool,
        include_historical: bool,
    ) -> List[Dict[str, Any]]:
        api_key = self.api_key or ODDS_API_KEY
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        async def run_fetch(allow_insecure: bool) -> List[Dict[str, Any]]:
            connector = self._build_connector(allow_insecure)
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                events_meta_frames: List[pd.DataFrame] = []

                event_index = await odds_fetch_event_index(
                    session,
                    api_key=api_key,
                    regions=ODDS_GAME_REGIONS,
                    bookmakers=ODDS_BOOKMAKERS,
                    markets=ODDS_EVENT_MARKETS,
                    odds_format=ODDS_FORMAT,
                    allow_insecure_ssl=allow_insecure,
                )
                if not event_index.empty:
                    event_index["event_id"] = event_index["event_id"].astype(str)
                    events_meta_frames.append(event_index)

                game_df = await odds_fetch_game_odds(
                    session,
                    api_key=api_key,
                    regions=ODDS_GAME_REGIONS,
                    markets=ODDS_DEFAULT_MARKETS,
                    bookmakers=ODDS_BOOKMAKERS,
                    odds_format=ODDS_FORMAT,
                    allow_insecure_ssl=allow_insecure,
                )
                if not game_df.empty:
                    game_df["event_id"] = game_df["event_id"].astype(str)
                    game_df['commence_dt'] = pd.to_datetime(
                        game_df['commence_time'], errors='coerce', utc=True
                    )
                    start_bound = self._normalize_bound(start)
                    end_bound = self._normalize_bound(end)
                    if start_bound is not None:
                        game_df = game_df[game_df['commence_dt'] >= start_bound]
                    if end_bound is not None:
                        game_df = game_df[game_df['commence_dt'] <= end_bound]
                if include_historical:
                    history_events = await odds_fetch_events_range(
                        session,
                        api_key=api_key,
                        start=start,
                        end=end,
                        regions=ODDS_GAME_REGIONS,
                        bookmakers=ODDS_BOOKMAKERS,
                        markets=ODDS_EVENT_MARKETS,
                        odds_format=ODDS_FORMAT,
                        allow_insecure_ssl=allow_insecure,
                    )
                    if not history_events.empty:
                        history_events['event_id'] = history_events['event_id'].astype(str)
                        events_meta_frames.append(history_events)

                events_meta = pd.DataFrame()
                if events_meta_frames:
                    events_meta = pd.concat(events_meta_frames, ignore_index=True)
                    if 'event_id' in events_meta.columns:
                        events_meta['event_id'] = events_meta['event_id'].astype(str)
                    else:
                        events_meta['event_id'] = ''
                    events_meta = events_meta[events_meta['event_id'].astype(bool)]
                    events_meta = events_meta.drop_duplicates(subset=['event_id'], keep='last')
                    if 'commence_dt' not in events_meta.columns:
                        events_meta['commence_dt'] = pd.to_datetime(
                            events_meta.get('commence_time'), errors='coerce', utc=True
                        )
                    else:
                        events_meta['commence_dt'] = pd.to_datetime(
                            events_meta['commence_dt'], errors='coerce', utc=True
                        )
                    start_bound = self._normalize_bound(start)
                    end_bound = self._normalize_bound(end)
                    if start_bound is not None:
                        events_meta = events_meta[
                            events_meta['commence_dt'].isna()
                            | (events_meta['commence_dt'] >= start_bound)
                        ]
                    if end_bound is not None:
                        events_meta = events_meta[
                            events_meta['commence_dt'].isna()
                            | (events_meta['commence_dt'] <= end_bound)
                        ]

                if not game_df.empty and not events_meta.empty:
                    merge_columns = ['event_id', 'home_team', 'away_team', 'commence_time', 'commence_dt']
                    available_cols = [col for col in merge_columns if col in events_meta.columns]
                    if available_cols:
                        game_df = game_df.merge(
                            events_meta[available_cols],
                            on='event_id',
                            how='left',
                            suffixes=('', '_meta'),
                        )
                        for column in ['home_team', 'away_team', 'commence_time', 'commence_dt']:
                            meta_col = f"{column}_meta"
                            if meta_col in game_df.columns:
                                game_df[column] = game_df[column].fillna(game_df[meta_col])
                                game_df.drop(columns=[meta_col], inplace=True)
                        if 'commence_time' in game_df.columns:
                            game_df['commence_dt'] = pd.to_datetime(
                                game_df['commence_time'], errors='coerce', utc=True
                            )
                        events_meta['commence_dt'] = pd.to_datetime(
                            events_meta['commence_time'], errors='coerce', utc=True
                        )
                        existing_ids: Set[str] = set()
                        if not game_df.empty and 'event_id' in game_df.columns:
                            existing_ids = set(game_df['event_id'].dropna().astype(str))
                        hist_event_ids = [
                            eid for eid in events_meta['event_id'] if eid and eid not in existing_ids
                        ]
                        if hist_event_ids:
                            history_games = await odds_fetch_event_game_markets(
                                session,
                                api_key=api_key,
                                event_ids=hist_event_ids,
                                regions=ODDS_GAME_REGIONS,
                                markets=ODDS_DEFAULT_MARKETS,
                                bookmakers=ODDS_BOOKMAKERS,
                                odds_format=ODDS_FORMAT,
                                allow_insecure_ssl=allow_insecure,
                                fallback_to_history=True,
                            )
                            if not history_games.empty:
                                history_games = history_games.merge(
                                    events_meta[
                                        [
                                            'event_id',
                                            'home_team',
                                            'away_team',
                                            'commence_time',
                                            'commence_dt',
                                        ]
                                    ],
                                    on='event_id',
                                    how='left',
                                    suffixes=('', '_meta'),
                                )
                                for column in ['home_team', 'away_team', 'commence_time', 'commence_dt']:
                                    meta_col = f"{column}_meta"
                                    if meta_col in history_games.columns:
                                        history_games[column] = history_games[column].fillna(
                                            history_games[meta_col]
                                        )
                                        history_games.drop(columns=[meta_col], inplace=True)
                                if 'commence_dt' in history_games.columns:
                                    history_games['commence_dt'] = pd.to_datetime(
                                        history_games['commence_dt'], errors='coerce', utc=True
                                    )
                                game_df = (
                                    pd.concat([game_df, history_games], ignore_index=True)
                                    if not game_df.empty
                                    else history_games
                                )
                if not game_df.empty:
                    start_bound = self._normalize_bound(start)
                    end_bound = self._normalize_bound(end)
                    if start_bound is not None:
                        game_df = game_df[game_df['commence_dt'] >= start_bound]
                    if end_bound is not None:
                        game_df = game_df[game_df['commence_dt'] <= end_bound]
                prop_df = pd.DataFrame()
                if include_player_props:
                    event_ids: List[str] = []
                    if not events_meta.empty and 'event_id' in events_meta.columns:
                        event_ids = (
                            events_meta['event_id']
                            .dropna()
                            .astype(str)
                            .drop_duplicates()
                            .tolist()
                        )
                    elif not game_df.empty and 'event_id' in game_df.columns:
                        event_ids = (
                            game_df['event_id']
                            .dropna()
                            .astype(str)
                            .drop_duplicates()
                            .tolist()
                        )
                    if event_ids:
                        prop_df = await odds_fetch_prop_odds(
                            session,
                            api_key=api_key,
                            event_ids=event_ids,
                            regions=ODDS_PROP_REGIONS,
                            bookmakers=ODDS_BOOKMAKERS,
                            odds_format=ODDS_FORMAT,
                            allow_insecure_ssl=allow_insecure,
                            fallback_to_history=True,
                        )
                        if not prop_df.empty:
                            if not events_meta.empty:
                                merge_columns = ['event_id', 'home_team', 'away_team', 'commence_time', 'commence_dt']
                                available_cols = [
                                    col for col in merge_columns if col in events_meta.columns
                                ]
                                if available_cols:
                                    prop_df = prop_df.merge(
                                        events_meta[available_cols],
                                        on='event_id',
                                        how='left',
                                        suffixes=('', '_meta'),
                                    )
                                    for column in [
                                        'home_team',
                                        'away_team',
                                        'commence_time',
                                        'commence_dt',
                                    ]:
                                        meta_col = f"{column}_meta"
                                        if meta_col in prop_df.columns:
                                            prop_df[column] = prop_df[column].fillna(prop_df[meta_col])
                                            prop_df.drop(columns=[meta_col], inplace=True)
                            prop_df['commence_dt'] = pd.to_datetime(
                                prop_df['commence_time'], errors='coerce', utc=True
                            )
                            start_bound = self._normalize_bound(start)
                            end_bound = self._normalize_bound(end)
                            if start_bound is not None:
                                prop_df = prop_df[prop_df['commence_dt'] >= start_bound]
                            if end_bound is not None:
                                prop_df = prop_df[prop_df['commence_dt'] <= end_bound]
                    if include_historical and not events_meta.empty:
                        hist_prop_ids = [
                            eid
                            for eid in events_meta['event_id']
                            if eid and (prop_df.empty or eid not in set(prop_df['event_id'].astype(str)))
                        ]
                        if hist_prop_ids:
                            history_props = await odds_fetch_prop_odds(
                                session,
                                api_key=api_key,
                                event_ids=hist_prop_ids,
                                regions=ODDS_PROP_REGIONS,
                                bookmakers=ODDS_BOOKMAKERS,
                                odds_format=ODDS_FORMAT,
                                allow_insecure_ssl=allow_insecure,
                                fallback_to_history=True,
                            )
                            if not history_props.empty:
                                history_props = history_props.merge(
                                    events_meta[
                                        [
                                            'event_id',
                                            'home_team',
                                            'away_team',
                                            'commence_time',
                                            'commence_dt',
                                        ]
                                    ],
                                    on='event_id',
                                    how='left',
                                    suffixes=('', '_meta'),
                                )
                                for column in ['home_team', 'away_team', 'commence_time', 'commence_dt']:
                                    meta_col = f"{column}_meta"
                                    if meta_col in history_props.columns:
                                        history_props[column] = history_props[column].fillna(
                                            history_props[meta_col]
                                        )
                                        history_props.drop(columns=[meta_col], inplace=True)
                                if 'commence_time' in history_props.columns:
                                    history_props['commence_dt'] = pd.to_datetime(
                                        history_props['commence_time'], errors='coerce', utc=True
                                    )
                                if start_bound is not None:
                                    history_props = history_props[
                                        history_props['commence_dt'] >= start_bound
                                    ]
                                if end_bound is not None:
                                    history_props = history_props[
                                        history_props['commence_dt'] <= end_bound
                                    ]
                                prop_df = (
                                    pd.concat([prop_df, history_props], ignore_index=True)
                                    if not prop_df.empty
                                    else history_props
                                )
                if not prop_df.empty and 'commence_dt' in prop_df.columns:
                    start_bound = self._normalize_bound(start)
                    end_bound = self._normalize_bound(end)
                    if start_bound is not None:
                        prop_df = prop_df[prop_df['commence_dt'] >= start_bound]
                    if end_bound is not None:
                        prop_df = prop_df[prop_df['commence_dt'] <= end_bound]
                return self._assemble_events(events_meta, game_df, prop_df, start, end)

        try:
            return await run_fetch(self.allow_insecure_ssl)
        except client_exceptions.ClientConnectorCertificateError:
            if self.allow_insecure_ssl:
                raise
            odds_logger.warning(
                "SSL verification failed when fetching odds; retrying with certificate verification disabled"
            )
            return await run_fetch(True)

    def fetch_odds(
        self,
        *,
        start: Optional[dt.datetime] = None,
        end: Optional[dt.datetime] = None,
        include_player_props: bool = True,
        include_historical: bool = True,
    ) -> List[Dict[str, Any]]:
        try:
            return asyncio.run(
                self._fetch_async(
                    start=start,
                    end=end,
                    include_player_props=include_player_props,
                    include_historical=include_historical,
                )
            )
        except RuntimeError:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return loop.run_until_complete(
                    self._fetch_async(
                        start=start,
                        end=end,
                        include_player_props=include_player_props,
                        include_historical=include_historical,
                    )
                )
            raise


@dataclass
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
        advanced_rows_map: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
        lineup_depth_teams: Set[str] = set()

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

                lineup_rows = self._lineup_rows_from_msf(
                    start_time,
                    away_team_abbr,
                    home_team_abbr,
                    self._msf_creds,
                    lineup_cache,
                )
                for lineup_row in lineup_rows:
                    if lineup_row.get("game_start") is None:
                        lineup_row["game_start"] = start_time

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
                        "offense_yards_per_play": self._safe_float(
                            advanced_payload.get("offense_yards_per_play")
                        ),
                        "defense_yards_per_play": self._safe_float(
                            advanced_payload.get("defense_yards_per_play")
                        ),
                        "offense_td_rate": self._safe_float(
                            advanced_payload.get("offense_td_rate")
                        ),
                        "defense_td_rate": self._safe_float(
                            advanced_payload.get("defense_td_rate")
                        ),
                        "pass_rate": self._safe_float(advanced_payload.get("pass_rate")),
                        "rush_rate": self._safe_float(advanced_payload.get("rush_rate")),
                        "pass_rate_over_expectation": self._safe_float(
                            advanced_payload.get("pass_rate_over_expectation")
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
        supplemental_rows = list(advanced_rows_map.values())
        if supplemental_rows:
            self.db.upsert_rows(
                self.db.team_advanced_metrics,
                supplemental_rows,
                ["season", "week", "team"],
            )

        skip_keys: Set[Tuple[str, int, str]] = set(
            (row.get("season"), int(row.get("week") or 0), row.get("team"))
            for row in supplemental_rows
            if row.get("team")
        )
        skip_keys.update(self.db.fetch_existing_advanced_metric_keys())
        derived_rows = self._derive_advanced_metrics_from_player_stats(skip_keys)
        if derived_rows:
            self.db.upsert_rows(
                self.db.team_advanced_metrics,
                derived_rows,
                ["season", "week", "team"],
            )

        # Ingest odds separately as they change frequently (always upsert)
        self._ingest_odds()

    def _ingest_odds(self) -> None:
        game_lookup_rows = self.db.fetch_game_lookup()
        if not game_lookup_rows:
            logging.warning("No scheduled games available when attempting to ingest odds.")
            return

        def _ensure_datetime(value: Any) -> Optional[dt.datetime]:
            if value is None or value == "":
                return None
            if isinstance(value, dt.datetime):
                return value
            return parse_dt(value)

        games_by_event: Dict[str, Dict[str, Any]] = {}
        seen_events: Set[str] = set()
        games_by_key: Dict[Tuple[str, str, Optional[dt.date]], List[Dict[str, Any]]] = defaultdict(list)
        refresh_candidates: Dict[str, Dict[str, Any]] = {}
        missing_count = 0
        stale_count = 0
        now_utc = default_now_utc()
        stale_cutoff = now_utc - dt.timedelta(hours=6)
        upcoming_window_end = now_utc + dt.timedelta(days=7)
        recent_window_start = now_utc - dt.timedelta(days=14)

        def _needs_refresh(row: Dict[str, Any], start_time: Optional[dt.datetime]) -> bool:
            nonlocal missing_count, stale_count
            home_price = row.get("home_moneyline")
            away_price = row.get("away_moneyline")
            odds_updated = _ensure_datetime(row.get("odds_updated"))

            missing_prices = pd.isna(home_price) or pd.isna(away_price)
            if missing_prices:
                missing_count += 1
                return True

            if start_time is None:
                return False

            is_recent_game = recent_window_start <= start_time <= upcoming_window_end
            if not is_recent_game:
                return False

            stale = odds_updated is None or odds_updated < stale_cutoff
            if stale:
                stale_count += 1
            return stale

        for row in game_lookup_rows:
            start_time = _ensure_datetime(row.get("start_time"))
            home = normalize_team_abbr(row.get("home_team"))
            away = normalize_team_abbr(row.get("away_team"))
            if home and away:
                key = (home, away, start_time.date() if start_time else None)
                games_by_key[key].append(row)
            event_id = row.get("odds_event_id")
            if event_id:
                games_by_event[str(event_id)] = row

            game_id = str(row.get("game_id")) if row.get("game_id") is not None else None
            if game_id:
                if _needs_refresh(row, start_time):
                    refresh_candidates[game_id] = row
                elif (
                    start_time is not None
                    and recent_window_start <= start_time <= upcoming_window_end
                ):
                    # Always refresh the upcoming slate to capture line movement and props
                    refresh_candidates.setdefault(game_id, row)

        rows_needing_refresh = list(refresh_candidates.values())

        if rows_needing_refresh:
            start_candidates = [
                _ensure_datetime(row.get("start_time"))
                for row in rows_needing_refresh
                if _ensure_datetime(row.get("start_time")) is not None
            ]
            if start_candidates:
                min_start = min(start_candidates)
                max_start = max(start_candidates)
            else:
                min_start = now_utc - dt.timedelta(days=7)
                max_start = now_utc + dt.timedelta(days=7)
        else:
            min_start = now_utc - dt.timedelta(days=7)
            max_start = now_utc + dt.timedelta(days=7)

        if not rows_needing_refresh:
            logging.info("All tracked games already have up-to-date sportsbook odds.")
            return

        logging.info(
            "Refreshing odds for %d games (missing=%d, stale=%d)",
            len(rows_needing_refresh),
            missing_count,
            stale_count,
        )

        if min_start is None:
            min_start = default_now_utc() - dt.timedelta(days=7)
        if max_start is None:
            max_start = default_now_utc() + dt.timedelta(days=7)

        buffer = dt.timedelta(days=2)
        odds_data = self.odds_client.fetch_odds(
            start=min_start - buffer,
            end=max_start + buffer,
            include_player_props=True,
            include_historical=True,
        )
        logging.info("Fetched %d odds entries", len(odds_data))

        historical_rows = [
            row
            for row in rows_needing_refresh
            if (
                _ensure_datetime(row.get("start_time")) is not None
                and _ensure_datetime(row.get("start_time")) < recent_window_start
            )
        ]
        if historical_rows:
            grouped_hist: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for hist_row in historical_rows:
                start_dt = _ensure_datetime(hist_row.get("start_time"))
                season_key = str(hist_row.get("season") or (start_dt.year if start_dt else "unknown"))
                grouped_hist[season_key].append(hist_row)

            history_buffer = dt.timedelta(days=10)
            chunk_span = dt.timedelta(days=45)
            for season_key, season_rows in grouped_hist.items():
                times = [
                    _ensure_datetime(row.get("start_time"))
                    for row in season_rows
                    if _ensure_datetime(row.get("start_time")) is not None
                ]
                if not times:
                    continue
                season_start = min(times) - history_buffer
                season_end = max(times) + history_buffer
                logging.info(
                    "Backfilling odds history for season %s (%d games)",
                    season_key,
                    len(season_rows),
                )
                current_start = season_start
                while current_start <= season_end:
                    current_end = min(current_start + chunk_span, season_end)
                    logging.debug(
                        "Historical odds window %s -> %s for season %s",
                        current_start.isoformat(),
                        current_end.isoformat(),
                        season_key,
                    )
                    extra_data = self.odds_client.fetch_odds(
                        start=current_start,
                        end=current_end,
                        include_player_props=True,
                        include_historical=True,
                    )
                    if extra_data:
                        logging.info(
                            "Fetched %d additional odds entries for season %s window %s -> %s",
                            len(extra_data),
                            season_key,
                            current_start.date(),
                            current_end.date(),
                        )
                        odds_data.extend(extra_data)
                    current_start = current_end + dt.timedelta(days=1)

        def _match_game(
            event_id: str,
            home_team: str,
            away_team: str,
            commence: Optional[dt.datetime],
        ) -> Optional[Dict[str, Any]]:
            if event_id in games_by_event:
                return games_by_event[event_id]

            date_candidates: Set[Optional[dt.date]] = {None}
            if commence is not None:
                base_date = commence.date()
                date_candidates = {
                    base_date,
                    (commence - dt.timedelta(days=1)).date(),
                    (commence + dt.timedelta(days=1)).date(),
                }

            candidates: List[Tuple[Dict[str, Any], float]] = []
            for date_value in date_candidates:
                key = (home_team, away_team, date_value)
                for record in games_by_key.get(key, []):
                    start_time = _ensure_datetime(record.get("start_time"))
                    if commence is not None and start_time is not None:
                        diff = abs((start_time - commence).total_seconds())
                    else:
                        diff = float("inf")
                    candidates.append((record, diff))

            if not candidates and commence is not None:
                # Try swapped home/away assignment if the API differs from MSF
                swapped_key_candidates: List[Tuple[Dict[str, Any], float]] = []
                for date_value in date_candidates:
                    key = (away_team, home_team, date_value)
                    for record in games_by_key.get(key, []):
                        start_time = _ensure_datetime(record.get("start_time"))
                        diff = (
                            abs((start_time - commence).total_seconds())
                            if start_time is not None
                            else float("inf")
                        )
                        swapped_key_candidates.append((record, diff))
                candidates.extend(swapped_key_candidates)

            if not candidates:
                return None

            candidates.sort(key=lambda item: item[1])
            matched_record = candidates[0][0]
            games_by_event[event_id] = matched_record
            return matched_record

        def _american_to_prob(odds: Optional[float]) -> Optional[float]:
            if odds is None:
                return None
            try:
                odds_val = float(odds)
            except (TypeError, ValueError):
                return None
            if odds_val > 0:
                return 100.0 / (odds_val + 100.0)
            return -odds_val / (-odds_val + 100.0)

        odds_rows: List[Dict[str, Any]] = []
        totals_rows: List[Dict[str, Any]] = []
        prop_rows: List[Dict[str, Any]] = []

        def _select_closing_bookmaker(
            bookmakers: Sequence[Dict[str, Any]],
            game_start: Optional[dt.datetime],
        ) -> Tuple[Optional[Dict[str, Any]], Optional[dt.datetime]]:
            if not bookmakers:
                return None, None

            ranked: List[Tuple[int, float, dt.datetime, Dict[str, Any]]] = []
            for bookmaker in bookmakers:
                update_raw = parse_dt(bookmaker.get("last_update"))
                update_ts = update_raw or default_now_utc()
                if game_start is not None:
                    if update_ts <= game_start:
                        priority = 0
                        delta = abs((game_start - update_ts).total_seconds())
                    elif update_ts <= game_start + dt.timedelta(minutes=15):
                        priority = 1
                        delta = abs((update_ts - game_start).total_seconds())
                    else:
                        priority = 2
                        delta = abs((update_ts - game_start).total_seconds())
                else:
                    priority = 2
                    delta = 0.0
                ranked.append((priority, float(delta), update_ts, bookmaker))

            ranked.sort(key=lambda item: (item[0], item[1], -item[2].timestamp()))
            best_priority, _, best_update, best_book = ranked[0]
            if best_priority >= 2 and bookmakers:
                # If everything is post-kickoff, still return the freshest quote.
                freshest = max(ranked, key=lambda item: item[2])
                best_update, best_book = freshest[2], freshest[3]
            return best_book, best_update

        for event in odds_data:
            event_id = str(event.get("id") or "")
            if not event_id or event_id in seen_events:
                continue
            seen_events.add(event_id)

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
                    event_id,
                    home_team_raw,
                    away_team_raw,
                )
                continue

            matched_game = _match_game(event_id, home_team, away_team, commence_time)
            if not matched_game:
                logging.debug(
                    "Unable to match odds event %s (%s at %s) to an existing MSF game", 
                    event_id,
                    away_team,
                    home_team,
                )
                continue

            game_id = matched_game.get("game_id")
            if not game_id:
                continue

            season = matched_game.get("season") or self._infer_season(commence_time)
            week = matched_game.get("week")
            markets = event.get("bookmakers", []) or []
            if not markets:
                continue

            # Choose bookmaker with the most recent update for moneylines
            sorted_books = sorted(
                markets,
                key=lambda b: parse_dt(b.get("last_update")) or default_now_utc(),
                reverse=True,
            )

            primary_book = sorted_books[0]
            last_update = parse_dt(primary_book.get("last_update")) or default_now_utc()
            moneyline_market = None
            for market in primary_book.get("markets", []) or []:
                if (market.get("key") or "").lower() == "h2h":
                    moneyline_market = market
                    break

            game_start = commence_time or _ensure_datetime(matched_game.get("start_time"))
            closing_book, closing_time = _select_closing_bookmaker(sorted_books, game_start)
            closing_market = None
            if closing_book is not None:
                for market in closing_book.get("markets", []) or []:
                    if (market.get("key") or "").lower() == "h2h":
                        closing_market = market
                        break

            def _extract_team_price(outcomes: Iterable[Dict[str, Any]], team_abbr: str) -> Optional[float]:
                for outcome in outcomes:
                    name_norm = normalize_team_abbr(outcome.get("name"))
                    desc_norm = normalize_team_abbr(outcome.get("description"))
                    if team_abbr in {name_norm, desc_norm}:
                        price = outcome.get("price")
                        try:
                            return float(price)
                        except (TypeError, ValueError):
                            return None
                return None

            home_price = None
            away_price = None
            if moneyline_market:
                outcomes = moneyline_market.get("outcomes", []) or []
                home_price = _extract_team_price(outcomes, home_team)
                away_price = _extract_team_price(outcomes, away_team)

            closing_home = None
            closing_away = None
            if closing_market:
                outcomes = closing_market.get("outcomes", []) or []
                closing_home = _extract_team_price(outcomes, home_team)
                closing_away = _extract_team_price(outcomes, away_team)

            closing_book_name = None
            if closing_book is not None:
                closing_book_name = closing_book.get("key") or closing_book.get("title") or "unknown"

            odds_rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "start_time": game_start,
                    "home_team": home_team,
                    "away_team": away_team,
                    "status": matched_game.get("status") or "scheduled",
                    "home_moneyline": home_price,
                    "away_moneyline": away_price,
                    "home_implied_prob": _american_to_prob(home_price),
                    "away_implied_prob": _american_to_prob(away_price),
                    "home_closing_moneyline": closing_home,
                    "away_closing_moneyline": closing_away,
                    "home_closing_implied_prob": _american_to_prob(closing_home),
                    "away_closing_implied_prob": _american_to_prob(closing_away),
                    "closing_bookmaker": closing_book_name,
                    "closing_line_time": closing_time,
                    "odds_updated": last_update,
                    "odds_event_id": event_id,
                }
            )

            for bookmaker in sorted_books:
                sportsbook = bookmaker.get("key") or bookmaker.get("title") or "unknown"
                book_update = parse_dt(bookmaker.get("last_update")) or last_update
                for market in bookmaker.get("markets", []) or []:
                    market_key = (market.get("key") or "").lower()
                    outcomes = market.get("outcomes", []) or []

                    if market_key == "totals":
                        over_outcome = next(
                            (o for o in outcomes if str(o.get("name", "")).lower() == "over"),
                            None,
                        )
                        under_outcome = next(
                            (o for o in outcomes if str(o.get("name", "")).lower() == "under"),
                            None,
                        )
                        try:
                            total_line = float(
                                (over_outcome or under_outcome or {}).get("point")
                                or (over_outcome or under_outcome or {}).get("line")
                            )
                        except (TypeError, ValueError):
                            total_line = None
                        if total_line is not None:
                            try:
                                over_price_val = float(over_outcome.get("price")) if over_outcome else None
                            except (TypeError, ValueError):
                                over_price_val = None
                            try:
                                under_price_val = float(under_outcome.get("price")) if under_outcome else None
                            except (TypeError, ValueError):
                                under_price_val = None
                            totals_rows.append(
                                {
                                    "market_id": f"{game_id}:{sportsbook}:total",
                                    "game_id": game_id,
                                    "event_id": event_id,
                                    "season": season,
                                    "week": week,
                                    "start_time": commence_time,
                                    "bookmaker": sportsbook,
                                    "total_line": total_line,
                                    "over_odds": over_price_val,
                                    "under_odds": under_price_val,
                                    "last_update": book_update,
                                }
                            )

                    elif canonical_prop_market_key(market_key) in ODDS_PLAYER_PROP_MARKETS:
                        normalized_market_key = canonical_prop_market_key(market_key)
                        player_buckets: Dict[str, Dict[str, Any]] = {}

                        for outcome in outcomes:
                            name_raw = str(outcome.get("name") or "").strip()
                            desc_raw = str(outcome.get("description") or "").strip()
                            participant = str(
                                outcome.get("participant")
                                or outcome.get("player")
                                or outcome.get("player_name")
                                or ""
                            ).strip()
                            side: Optional[str] = None
                            player_name: Optional[str] = None

                            lower_name = name_raw.lower()
                            lower_desc = desc_raw.lower()
                            if lower_name in {"over", "under"}:
                                side = name_raw.title()
                                player_name = desc_raw or participant
                            elif lower_desc in {"over", "under"}:
                                side = desc_raw.title()
                                player_name = name_raw or participant
                            elif lower_name.endswith(" over") or lower_name.endswith(" under"):
                                tokens = lower_name.rsplit(" ", 1)
                                side = tokens[1].title()
                                player_name = name_raw[: -len(tokens[1])].strip()
                            elif participant:
                                player_name = participant

                            if not player_name or not side:
                                continue

                            player_line = outcome.get("line") or outcome.get("point")
                            try:
                                player_line_val = float(player_line) if player_line is not None else None
                            except (TypeError, ValueError):
                                player_line_val = None

                            price_val = outcome.get("price")
                            try:
                                price_float = float(price_val) if price_val is not None else None
                            except (TypeError, ValueError):
                                price_float = None

                            player_key = robust_player_name_key(player_name)
                            bucket = player_buckets.setdefault(
                                player_key,
                                {
                                    "player_name": player_name,
                                    "player_id": outcome.get("player_id")
                                    or outcome.get("participant_id")
                                    or outcome.get("id"),
                                    "line": player_line_val,
                                    "over_odds": None,
                                    "under_odds": None,
                                    "team": normalize_team_abbr(outcome.get("team")),
                                },
                            )

                            if player_line_val is not None:
                                bucket["line"] = player_line_val
                            if side == "Over":
                                bucket["over_odds"] = price_float
                            elif side == "Under":
                                bucket["under_odds"] = price_float

                        for player_key, info in player_buckets.items():
                            if info.get("line") is None:
                                continue
                            if info.get("over_odds") is None and info.get("under_odds") is None:
                                continue
                            player_name_norm = normalize_player_name(info.get("player_name"))
                            team_abbr = info.get("team")
                            opponent = None
                            if team_abbr:
                                if team_abbr == home_team:
                                    opponent = away_team
                                elif team_abbr == away_team:
                                    opponent = home_team
                            prop_rows.append(
                                {
                                    "prop_id": f"{game_id}:{sportsbook}:{normalized_market_key}:{player_key}",
                                    "game_id": game_id,
                                    "event_id": event_id,
                                    "player_id": str(info.get("player_id")) if info.get("player_id") else None,
                                    "player_name": info.get("player_name"),
                                    "player_name_norm": player_name_norm,
                                    "team": team_abbr,
                                    "opponent": opponent,
                                    "market": normalized_market_key,
                                    "line": info.get("line"),
                                    "over_odds": info.get("over_odds"),
                                    "under_odds": info.get("under_odds"),
                                    "bookmaker": sportsbook,
                                    "last_update": book_update,
                                }
                            )

        if odds_rows:
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
                    "home_closing_moneyline",
                    "away_closing_moneyline",
                    "home_closing_implied_prob",
                    "away_closing_implied_prob",
                    "closing_bookmaker",
                    "closing_line_time",
                    "odds_updated",
                    "home_team",
                    "away_team",
                    "odds_event_id",
                    "season",
                    "week",
                ],
            )

        if totals_rows:
            self.db.upsert_rows(
                self.db.game_totals,
                totals_rows,
                ["market_id"],
                update_columns=[
                    "total_line",
                    "over_odds",
                    "under_odds",
                    "last_update",
                    "start_time",
                    "season",
                    "week",
                    "bookmaker",
                    "event_id",
                ],
            )

        if prop_rows:
            self.db.upsert_rows(
                self.db.player_prop_lines,
                prop_rows,
                ["prop_id"],
                update_columns=[
                    "line",
                    "over_odds",
                    "under_odds",
                    "last_update",
                    "team",
                    "opponent",
                    "bookmaker",
                    "event_id",
                    "player_id",
                ],
            )

    def _derive_advanced_metrics_from_player_stats(
        self, skip_keys: Set[Tuple[str, int, str]]
    ) -> List[Dict[str, Any]]:
        try:
            player_stats = pd.read_sql_table("nfl_player_stats", self.db.engine)
        except Exception:
            logging.exception(
                "Failed to load player stats for derived advanced metric generation"
            )
            return []

        if player_stats.empty:
            return []

        try:
            games = pd.read_sql_table(
                "nfl_games",
                self.db.engine,
                columns=["game_id", "season", "week", "home_team", "away_team"],
            )
        except Exception:
            logging.exception(
                "Failed to load games metadata for derived advanced metric generation"
            )
            return []

        if games.empty or "game_id" not in games.columns:
            return []

        enrichment_cols = [
            "rushing_attempts",
            "rushing_yards",
            "rushing_tds",
            "receiving_targets",
            "receiving_yards",
            "receiving_tds",
            "passing_attempts",
            "passing_yards",
            "passing_tds",
        ]
        for col in enrichment_cols:
            if col not in player_stats.columns:
                player_stats[col] = 0.0

        merged = player_stats.merge(games, on="game_id", how="left")
        merged = merged.dropna(subset=["season", "week", "team"])
        if merged.empty:
            return []

        merged["season"] = merged["season"].astype(str)
        merged["week"] = merged["week"].apply(lambda x: int(x) if pd.notna(x) else None)
        merged = merged.dropna(subset=["week"])
        if merged.empty:
            return []

        for col in ("team", "home_team", "away_team"):
            if col in merged.columns:
                merged[col] = merged[col].apply(normalize_team_abbr)

        derived = FeatureBuilder._compute_team_unit_strength(merged, None)
        if derived is None or derived.empty:
            return []

        derived = derived.replace({np.nan: None})
        metrics_columns = [
            "pace_seconds_per_play",
            "offense_epa",
            "defense_epa",
            "offense_success_rate",
            "defense_success_rate",
            "offense_yards_per_play",
            "defense_yards_per_play",
            "offense_td_rate",
            "defense_td_rate",
            "pass_rate",
            "rush_rate",
            "pass_rate_over_expectation",
            "travel_penalty",
            "rest_penalty",
            "weather_adjustment",
        ]

        rows: List[Dict[str, Any]] = []
        for _, metric_row in derived.iterrows():
            season = str(metric_row.get("season"))
            week_val = metric_row.get("week")
            team = normalize_team_abbr(metric_row.get("team"))
            if not team:
                continue
            if week_val is None or pd.isna(week_val):
                continue
            week_int = int(week_val)
            key = (season, week_int, team)
            if key in skip_keys:
                continue
            row: Dict[str, Any] = {
                "metric_id": f"derived_{season}_{week_int}_{team}",
                "season": season,
                "week": week_int,
                "team": team,
            }
            for col in metrics_columns:
                row[col] = self._safe_float(metric_row.get(col))
            rows.append(row)

        if rows:
            logging.info("Derived %d advanced metric rows from player stats", len(rows))
        return rows

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

        collected_by_team: Dict[str, List[Dict[str, Any]]] = {}
        found_payload = False

        for date_key in date_candidates:
            for lineup_type in (None, "expected"):
                cache_token = f"{season_slug}|{date_key}|{lineup_type or 'default'}"
                cache_key = (cache_token, away_norm, home_norm)
                if cache_key in lineup_cache:
                    cached = lineup_cache[cache_key]
                    if cached:
                        for rec in cached:
                            collected_by_team.setdefault(rec.get("team"), []).append(rec)
                        found_payload = True
                        break
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
                    found_payload = True
                    for rec in enriched_rows:
                        collected_by_team.setdefault(rec.get("team"), []).append(rec)
                    break
            if found_payload:
                break

        def _lineup_needs_team(team_code: str) -> bool:
            team_rows = collected_by_team.get(team_code) or []
            if not team_rows:
                return True
            if not any(normalize_position(r.get("base_pos")) == "QB" for r in team_rows):
                return True
            for pos_key, max_count in _OFFENSE_KEEP.items():
                pos_rows = [
                    r
                    for r in team_rows
                    if normalize_position(r.get("base_pos")) == pos_key
                ]
                if not pos_rows:
                    return True
                if len(pos_rows) < max_count:
                    return True
            return False

        def _merge_records(existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            if not incoming:
                return existing
            merged = (existing or []) + incoming

            def priority(rec: Dict[str, Any]) -> Tuple[int, int]:
                source = rec.get("source") or ""
                source_rank = 0 if str(source).startswith("msf-lineup") else 1
                depth_val = rec.get("rank")
                if depth_val is None:
                    depth_val = 99
                return (source_rank, depth_val)

            deduped: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for rec in sorted(merged, key=priority):
                team = rec.get("team")
                pos = normalize_position(rec.get("base_pos"))
                key = (team, f"{pos}:{rec.get('__pname_key')}")
                if key not in deduped:
                    deduped[key] = rec
            return list(deduped.values())

        def _fetch_team_depth(team_code: str) -> List[Dict[str, Any]]:
            team_rows: List[Dict[str, Any]] = []
            team_cache_prefix = f"{season_slug}|team|{team_code}"
            for lineup_type in (None, "expected"):
                cache_token = f"{team_cache_prefix}|{lineup_type or 'default'}"
                cache_key = (cache_token, team_code, team_code)
                if cache_key in lineup_cache:
                    cached_rows = lineup_cache[cache_key]
                    if cached_rows:
                        team_rows = cached_rows
                        break
                    continue
                url = (
                    f"https://api.mysportsfeeds.com/v2.1/pull/nfl/{season_slug}/teams/"
                    f"{team_code}/lineup.json"
                )
                params = {"lineupType": lineup_type} if lineup_type else None
                response = _http_get_with_retry(
                    url,
                    auth,
                    params=params,
                    headers=accept_headers,
                )
                if response is None:
                    lineup_cache[cache_key] = []
                    continue
                if response.status_code in {401, 404}:
                    lineup_cache[cache_key] = []
                    if response.status_code == 401:
                        logging.warning(
                            "lineup: 401 unauthorized for %s (check MSF credentials)",
                            url,
                        )
                        return []
                    continue
                if response.status_code == 204:
                    lineup_cache[cache_key] = []
                    continue
                if response.status_code != 200:
                    logging.info("lineup: %s returned %s", url, response.status_code)
                    lineup_cache[cache_key] = []
                    continue
                try:
                    payload = response.json()
                except Exception:
                    logging.exception("lineup: JSON decode failed for %s", url)
                    lineup_cache[cache_key] = []
                    continue
                rows = _extract_lineup_rows(payload if isinstance(payload, dict) else {})
                enriched: List[Dict[str, Any]] = []
                updated_at = (
                    parse_dt(payload.get("lastUpdatedOn"))
                    if isinstance(payload, dict)
                    else None
                )
                for rec in rows:
                    if _msf_team_abbr(rec.get("team")) != team_code:
                        continue
                    enriched.append(
                        {
                            "team": team_code,
                            "position": rec.get("position"),
                            "player_id": rec.get("player_id") or "",
                            "player_name": rec.get("player_name"),
                            "first_name": rec.get("first_name"),
                            "last_name": rec.get("last_name"),
                            "rank": rec.get("rank"),
                            "depth_id": (
                                f"msf-team-lineup:{team_code}:{rec.get('position')}:{rec.get('player_id') or rec.get('__pname_key')}"
                            ),
                            "updated_at": updated_at,
                            "source": "msf-team-lineup",
                            "player_team": rec.get("player_team"),
                            "game_start": start_dt,
                            "__pname_key": rec.get("__pname_key"),
                            "side": rec.get("side"),
                            "base_pos": rec.get("base_pos") or rec.get("position"),
                            "playing_probability": rec.get("playing_probability"),
                            "status_bucket": rec.get("status_bucket"),
                            "practice_status": rec.get("practice_status"),
                        }
                    )
                lineup_cache[cache_key] = enriched
                if enriched:
                    team_rows = enriched
                    break
            return team_rows

        for team_code in (away_norm, home_norm):
            if _lineup_needs_team(team_code):
                supplemental = _fetch_team_depth(team_code)
                if supplemental:
                    collected_by_team[team_code] = _merge_records(
                        collected_by_team.get(team_code, []), supplemental
                    )

        flattened: List[Dict[str, Any]] = []
        for team_code, team_rows in collected_by_team.items():
            if not team_rows:
                continue
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for rec in team_rows:
                pos = normalize_position(rec.get("base_pos"))
                if pos not in _OFFENSE_KEEP:
                    continue
                grouped.setdefault(pos, []).append(rec)
            for pos, recs in grouped.items():
                recs_sorted = sorted(
                    recs,
                    key=lambda r: (
                        0 if str(r.get("source", "")).startswith("msf-lineup") else 1,
                        r.get("rank") if r.get("rank") is not None else 99,
                        r.get("player_name") or r.get("__pname_key") or "",
                    ),
                )
                keep = _OFFENSE_KEEP.get(pos, 0)
                for rec in recs_sorted[:keep]:
                    if rec.get("game_start") is None:
                        rec["game_start"] = start_dt
                    flattened.append(rec)

        if flattened:
            return flattened

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

    def __init__(
        self, engine: Engine, supplemental_loader: Optional[SupplementalDataLoader] = None
    ):
        self.engine = engine
        self.games_frame: Optional[pd.DataFrame] = None
        self.games_frame_raw: Optional[pd.DataFrame] = None
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
        self.latest_odds_lookup: Optional[pd.DataFrame] = None
        self.game_totals_frame: Optional[pd.DataFrame] = None
        self.player_prop_lines_frame: Optional[pd.DataFrame] = None
        self.team_game_lookup: Optional[pd.DataFrame] = None
        self.supplemental_loader = supplemental_loader

    @staticmethod
    def _american_to_prob_series(series: pd.Series) -> pd.Series:
        odds = pd.to_numeric(series, errors="coerce")
        prob = pd.Series(np.nan, index=odds.index, dtype=float)
        if odds is None:
            return prob
        positive = odds >= 0
        prob.loc[positive] = 100.0 / (odds.loc[positive] + 100.0)
        prob.loc[~positive] = (-odds.loc[~positive]) / ((-odds.loc[~positive]) + 100.0)
        return prob

    @staticmethod
    def _merge_odds_snapshot(
        frame: pd.DataFrame,
        lookup: pd.DataFrame,
        key_cols: List[str],
        value_cols: List[str],
        sort_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if lookup is None or lookup.empty:
            return frame

        available_values = [col for col in value_cols if col in lookup.columns]
        if not available_values:
            return frame

        for col in key_cols:
            if col not in frame.columns or col not in lookup.columns:
                return frame

        subset = lookup.dropna(subset=[col for col in available_values if "moneyline" in col])
        if subset.empty:
            return frame

        ordering = sort_cols or []
        ordering = [col for col in ordering if col in subset.columns]
        if ordering:
            subset = subset.sort_values(ordering, ascending=[False] * len(ordering))
        subset = subset.drop_duplicates(subset=key_cols, keep="first")

        merge_cols = key_cols + available_values
        snapshot = subset[merge_cols]

        merged = frame.merge(snapshot, on=key_cols, how="left", suffixes=("", "_lookup"))
        for col in available_values:
            lookup_col = f"{col}_lookup"
            if lookup_col in merged.columns:
                merged[col] = merged[col].combine_first(merged[lookup_col])
                merged.drop(columns=[lookup_col], inplace=True)
        return merged

    @staticmethod
    def _augment_matchup_features(features: pd.DataFrame) -> pd.DataFrame:
        """Add relative matchup features (diffs/ratios) that help the models."""

        if features.empty:
            return features

        working = features.copy()

        def _maybe_diff(output: str, home_col: str, away_col: str) -> None:
            if home_col in working.columns and away_col in working.columns:
                working[output] = working[home_col] - working[away_col]

        diff_specs = {
            "offense_pass_rating_diff": (
                "home_offense_pass_rating",
                "away_offense_pass_rating",
            ),
            "offense_rush_rating_diff": (
                "home_offense_rush_rating",
                "away_offense_rush_rating",
            ),
            "defense_pass_rating_diff": (
                "home_defense_pass_rating",
                "away_defense_pass_rating",
            ),
            "defense_rush_rating_diff": (
                "home_defense_rush_rating",
                "away_defense_rush_rating",
            ),
            "offense_epa_diff": ("home_offense_epa", "away_offense_epa"),
            "defense_epa_diff": ("home_defense_epa", "away_defense_epa"),
            "offense_success_rate_diff": (
                "home_offense_success_rate",
                "away_offense_success_rate",
            ),
            "defense_success_rate_diff": (
                "home_defense_success_rate",
                "away_defense_success_rate",
            ),
            "offense_yards_per_play_diff": (
                "home_offense_yards_per_play",
                "away_offense_yards_per_play",
            ),
            "defense_yards_per_play_diff": (
                "home_defense_yards_per_play",
                "away_defense_yards_per_play",
            ),
            "offense_td_rate_diff": (
                "home_offense_td_rate",
                "away_offense_td_rate",
            ),
            "defense_td_rate_diff": (
                "home_defense_td_rate",
                "away_defense_td_rate",
            ),
            "pass_rate_diff": ("home_pass_rate", "away_pass_rate"),
            "rush_rate_diff": ("home_rush_rate", "away_rush_rate"),
            "pass_rate_over_expectation_diff": (
                "home_pass_rate_over_expectation",
                "away_pass_rate_over_expectation",
            ),
            "pace_seconds_diff": (
                "home_pace_seconds_per_play",
                "away_pace_seconds_per_play",
            ),
            "travel_penalty_diff": ("home_travel_penalty", "away_travel_penalty"),
            "rest_penalty_diff": ("home_rest_penalty", "away_rest_penalty"),
            "weather_adjustment_diff": (
                "home_weather_adjustment",
                "away_weather_adjustment",
            ),
            "timezone_diff_diff": (
                "home_timezone_diff_hours",
                "away_timezone_diff_hours",
            ),
            "points_for_avg_diff": (
                "home_points_for_avg",
                "away_points_for_avg",
            ),
            "points_against_avg_diff": (
                "home_points_against_avg",
                "away_points_against_avg",
            ),
            "point_diff_avg_diff": (
                "home_point_diff_avg",
                "away_point_diff_avg",
            ),
            "win_pct_recent_diff": (
                "home_win_pct_recent",
                "away_win_pct_recent",
            ),
            "rest_days_diff": ("home_rest_days", "away_rest_days"),
            "prev_points_for_diff": (
                "home_prev_points_for",
                "away_prev_points_for",
            ),
            "prev_points_against_diff": (
                "home_prev_points_against",
                "away_prev_points_against",
            ),
            "prev_point_diff_diff": (
                "home_prev_point_diff",
                "away_prev_point_diff",
            ),
            "elo_pre_diff": ("home_elo_pre", "away_elo_pre"),
            "elo_win_prob_diff": ("home_elo_win_prob", "away_elo_win_prob"),
            "elo_vs_opponent_diff": (
                "home_elo_vs_opponent",
                "away_elo_vs_opponent",
            ),
            "elo_edge_diff": ("home_elo_edge", "away_elo_edge"),
            "injury_total_diff": ("home_injury_total", "away_injury_total"),
        }

        for output, (home_col, away_col) in diff_specs.items():
            _maybe_diff(output, home_col, away_col)

        def _maybe_net(output: str, offense_col: str, defense_col: str) -> None:
            if offense_col in working.columns and defense_col in working.columns:
                working[output] = working[offense_col] - working[defense_col]

        net_specs = [
            ("home_net_epa", "home_offense_epa", "away_defense_epa"),
            ("away_net_epa", "away_offense_epa", "home_defense_epa"),
            (
                "home_net_success_rate",
                "home_offense_success_rate",
                "away_defense_success_rate",
            ),
            (
                "away_net_success_rate",
                "away_offense_success_rate",
                "home_defense_success_rate",
            ),
            (
                "home_net_yards_per_play",
                "home_offense_yards_per_play",
                "away_defense_yards_per_play",
            ),
            (
                "away_net_yards_per_play",
                "away_offense_yards_per_play",
                "home_defense_yards_per_play",
            ),
            (
                "home_net_td_rate",
                "home_offense_td_rate",
                "away_defense_td_rate",
            ),
            (
                "away_net_td_rate",
                "away_offense_td_rate",
                "home_defense_td_rate",
            ),
            (
                "home_pass_matchup",
                "home_offense_pass_rating",
                "away_defense_pass_rating",
            ),
            (
                "away_pass_matchup",
                "away_offense_pass_rating",
                "home_defense_pass_rating",
            ),
            (
                "home_rush_matchup",
                "home_offense_rush_rating",
                "away_defense_rush_rating",
            ),
            (
                "away_rush_matchup",
                "away_offense_rush_rating",
                "home_defense_rush_rating",
            ),
        ]

        for output, offense_col, defense_col in net_specs:
            _maybe_net(output, offense_col, defense_col)

        # Blend market-implied probabilities via logit difference for better calibration.
        if {
            "home_implied_prob",
            "away_implied_prob",
        }.issubset(working.columns):
            home_prob = pd.to_numeric(working["home_implied_prob"], errors="coerce")
            away_prob = pd.to_numeric(working["away_implied_prob"], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                home_logit = np.log((home_prob + 1e-6) / np.clip(1 - home_prob, 1e-6, None))
                away_logit = np.log((away_prob + 1e-6) / np.clip(1 - away_prob, 1e-6, None))
            working["implied_prob_logit_diff"] = home_logit - away_logit

        # Clip differential features to reduce the impact of extreme outliers. This uses
        # simple winsorization at the 1st/99th percentiles computed from the current frame.
        diff_columns = (
            list(diff_specs.keys())
            + [
                "implied_prob_logit_diff",
                "home_net_epa",
                "away_net_epa",
                "home_net_success_rate",
                "away_net_success_rate",
                "home_net_yards_per_play",
                "away_net_yards_per_play",
                "home_net_td_rate",
                "away_net_td_rate",
                "home_pass_matchup",
                "away_pass_matchup",
                "home_rush_matchup",
                "away_rush_matchup",
            ]
        )
        numeric_diff_columns = [col for col in diff_columns if col in working.columns]
        if numeric_diff_columns:
            diff_subset = working[numeric_diff_columns]
            if not diff_subset.empty:
                lower_bounds = diff_subset.quantile(0.01)
                upper_bounds = diff_subset.quantile(0.99)
                working[numeric_diff_columns] = diff_subset.clip(
                    lower=lower_bounds, upper=upper_bounds, axis=1
                )

        return working

    def _backfill_game_odds(self, games: pd.DataFrame) -> pd.DataFrame:
        if games is None or games.empty:
            return games

        working = games.copy()
        if "week" in working.columns:
            working["week"] = pd.to_numeric(working["week"], errors="coerce")
        if "start_time" in working.columns:
            working["start_time"] = pd.to_datetime(working["start_time"], errors="coerce")
            working["_start_date"] = working["start_time"].dt.normalize()

        odds_cols = [
            "home_moneyline",
            "away_moneyline",
            "home_implied_prob",
            "away_implied_prob",
            "home_closing_moneyline",
            "away_closing_moneyline",
            "home_closing_implied_prob",
            "away_closing_implied_prob",
            "closing_bookmaker",
            "closing_line_time",
            "odds_updated",
        ]

        lookup_source = working.dropna(subset=["home_moneyline", "away_moneyline"], how="any")
        if not lookup_source.empty:
            if "odds_updated" in lookup_source.columns:
                sort_columns = ["odds_updated", "start_time"]
            else:
                sort_columns = ["start_time"]

            key_sets: List[List[str]] = []
            if "game_id" in working.columns:
                key_sets.append(["game_id"])
            if {"season", "week", "home_team", "away_team"}.issubset(working.columns):
                key_sets.append(["season", "week", "home_team", "away_team"])
            if {"home_team", "away_team", "_start_date"}.issubset(working.columns):
                key_sets.append(["home_team", "away_team", "_start_date"])
            if {"season", "home_team", "away_team"}.issubset(working.columns):
                key_sets.append(["season", "home_team", "away_team"])

            for keys in key_sets:
                working = self._merge_odds_snapshot(
                    working, lookup_source, keys, odds_cols, sort_cols=sort_columns
                )

        for side in ("home", "away"):
            money_col = f"{side}_moneyline"
            prob_col = f"{side}_implied_prob"
            if money_col in working.columns:
                derived = self._american_to_prob_series(working[money_col])
                if prob_col in working.columns:
                    working[prob_col] = working[prob_col].combine_first(derived)
                else:
                    working[prob_col] = derived

            closing_money_col = f"{side}_closing_moneyline"
            closing_prob_col = f"{side}_closing_implied_prob"
            if closing_money_col in working.columns:
                derived_closing = self._american_to_prob_series(working[closing_money_col])
                if closing_prob_col in working.columns:
                    working[closing_prob_col] = working[closing_prob_col].combine_first(
                        derived_closing
                    )
                else:
                    working[closing_prob_col] = derived_closing

        loader = getattr(self, "supplemental_loader", None)
        supplemental_closing = None
        if loader is not None:
            supplemental_closing = getattr(loader, "closing_odds_frame", None)
        if supplemental_closing is not None and not supplemental_closing.empty:
            supplemental = supplemental_closing.copy()
            if "week" in supplemental.columns:
                supplemental["week"] = pd.to_numeric(supplemental["week"], errors="coerce")
            supplemental["season"] = supplemental["season"].astype(str)
            for col in ["home_team", "away_team"]:
                if col in supplemental.columns:
                    supplemental[col] = supplemental[col].apply(normalize_team_abbr)

            merge_keys: List[str] = []
            temporary_game_id_key = None

            if "game_id" in supplemental.columns and "game_id" in working.columns:
                # Avoid dtype mismatches (object vs. int64) when merging on game_id by
                # materializing a temporary, normalized key in both frames.
                temporary_game_id_key = "_merge_game_id"
                working[temporary_game_id_key] = working["game_id"].astype(str)
                supplemental[temporary_game_id_key] = supplemental["game_id"].astype(str)
                merge_keys = [temporary_game_id_key]
            else:
                season_team_keys = {"season", "home_team", "away_team"}
                working_has_core = season_team_keys.issubset(working.columns)
                supplemental_has_core = season_team_keys.issubset(supplemental.columns)
                supplemental_has_week = (
                    "week" in supplemental.columns
                    and supplemental["week"].notna().any()
                )
                if (
                    supplemental_has_week
                    and {"week"}.union(season_team_keys).issubset(supplemental.columns)
                    and {"week"}.union(season_team_keys).issubset(working.columns)
                ):
                    merge_keys = ["season", "week", "home_team", "away_team"]
                elif supplemental_has_core and working_has_core:
                    merge_keys = ["season", "home_team", "away_team"]

                if not merge_keys:
                    closing_time = pd.to_datetime(
                        supplemental.get("closing_line_time"), errors="coerce"
                    )
                    if closing_time.notna().any() and working_has_core:
                        supplemental["_merge_start_date"] = closing_time.dt.normalize()
                        working["_merge_start_date"] = working.get("_start_date")
                        if (
                            "_merge_start_date" in supplemental.columns
                            and "_merge_start_date" in working.columns
                        ):
                            merge_keys = [
                                "home_team",
                                "away_team",
                                "_merge_start_date",
                            ]

            if merge_keys:
                supplement_cols = [
                    "home_closing_moneyline",
                    "away_closing_moneyline",
                    "home_closing_implied_prob",
                    "away_closing_implied_prob",
                    "closing_bookmaker",
                    "closing_line_time",
                ]
                available_cols = [
                    col for col in supplement_cols if col in supplemental.columns
                ]
                if available_cols:
                    rename_map = {col: f"{col}_supp" for col in available_cols}
                    supplemental_subset = supplemental[merge_keys + available_cols].rename(
                        columns=rename_map
                    )
                    working = working.merge(
                        supplemental_subset, on=merge_keys, how="left"
                    )
                    for col in available_cols:
                        supplemental_col = f"{col}_supp"
                        if supplemental_col in working.columns:
                            working[col] = working[col].combine_first(
                                working[supplemental_col]
                            )
                            working.drop(columns=[supplemental_col], inplace=True)

                # If orientation is flipped (e.g., neutral-site listings) the direct merge above
                # will not populate closing lines. Build an order-agnostic team key so we can
                # realign supplemental rows to the schedule’s home/away designation.
                if available_cols:
                    def _team_pair(series: pd.Series) -> str:
                        teams = [
                            str(series.get("home_team", "") or "").strip(),
                            str(series.get("away_team", "") or "").strip(),
                        ]
                        teams = [team for team in teams if team]
                        return "|".join(sorted(teams)) if teams else ""

                    supplemental["_team_pair"] = supplemental.apply(_team_pair, axis=1)
                    working["_team_pair"] = working.apply(_team_pair, axis=1)

                    pair_keys: List[str] = []
                    if {
                        "season",
                        "week",
                        "_team_pair",
                    }.issubset(supplemental.columns) and {
                        "season",
                        "week",
                        "_team_pair",
                    }.issubset(working.columns):
                        pair_keys = ["season", "week", "_team_pair"]
                    elif {"season", "_team_pair"}.issubset(supplemental.columns) and {
                        "season",
                        "_team_pair",
                    }.issubset(working.columns):
                        pair_keys = ["season", "_team_pair"]
                    elif {
                        "_merge_start_date",
                        "_team_pair",
                    }.issubset(supplemental.columns) and {
                        "_merge_start_date",
                        "_team_pair",
                    }.issubset(working.columns):
                        pair_keys = ["_merge_start_date", "_team_pair"]

                    if pair_keys:
                        pair_subset = supplemental[
                            pair_keys
                            + ["home_team", "away_team"]
                            + available_cols
                        ].copy()
                        pair_subset = pair_subset.dropna(
                            subset=["home_team", "away_team", "_team_pair"]
                        )
                        pair_subset = pair_subset.rename(
                            columns={
                                "home_team": "home_team_pair",
                                "away_team": "away_team_pair",
                                **{col: f"{col}_pair" for col in available_cols},
                            }
                        )
                        pair_subset = pair_subset.drop_duplicates(subset=pair_keys)

                        working = working.merge(
                            pair_subset, on=pair_keys, how="left"
                        )

                        same_orientation = (
                            working.get("home_team_pair") == working.get("home_team")
                        )
                        swapped_orientation = (
                            working.get("home_team_pair") == working.get("away_team")
                        )

                        if "home_closing_moneyline_pair" in working.columns:
                            mask = (
                                working["home_closing_moneyline"].isna()
                                & same_orientation
                                & working["home_closing_moneyline_pair"].notna()
                            )
                            working.loc[
                                mask, "home_closing_moneyline"
                            ] = working.loc[mask, "home_closing_moneyline_pair"]

                            mask = (
                                working["home_closing_moneyline"].isna()
                                & swapped_orientation
                                & working["away_closing_moneyline_pair"].notna()
                            )
                            working.loc[
                                mask, "home_closing_moneyline"
                            ] = working.loc[mask, "away_closing_moneyline_pair"]

                        if "away_closing_moneyline_pair" in working.columns:
                            mask = (
                                working["away_closing_moneyline"].isna()
                                & same_orientation
                                & working["away_closing_moneyline_pair"].notna()
                            )
                            working.loc[
                                mask, "away_closing_moneyline"
                            ] = working.loc[mask, "away_closing_moneyline_pair"]

                            mask = (
                                working["away_closing_moneyline"].isna()
                                & swapped_orientation
                                & working["home_closing_moneyline_pair"].notna()
                            )
                            working.loc[
                                mask, "away_closing_moneyline"
                            ] = working.loc[mask, "home_closing_moneyline_pair"]

                        if "home_closing_implied_prob_pair" in working.columns:
                            mask = (
                                working["home_closing_implied_prob"].isna()
                                & same_orientation
                                & working["home_closing_implied_prob_pair"].notna()
                            )
                            working.loc[
                                mask, "home_closing_implied_prob"
                            ] = working.loc[mask, "home_closing_implied_prob_pair"]

                            mask = (
                                working["home_closing_implied_prob"].isna()
                                & swapped_orientation
                                & working["away_closing_implied_prob_pair"].notna()
                            )
                            working.loc[
                                mask, "home_closing_implied_prob"
                            ] = working.loc[mask, "away_closing_implied_prob_pair"]

                        if "away_closing_implied_prob_pair" in working.columns:
                            mask = (
                                working["away_closing_implied_prob"].isna()
                                & same_orientation
                                & working["away_closing_implied_prob_pair"].notna()
                            )
                            working.loc[
                                mask, "away_closing_implied_prob"
                            ] = working.loc[mask, "away_closing_implied_prob_pair"]

                            mask = (
                                working["away_closing_implied_prob"].isna()
                                & swapped_orientation
                                & working["home_closing_implied_prob_pair"].notna()
                            )
                            working.loc[
                                mask, "away_closing_implied_prob"
                            ] = working.loc[mask, "home_closing_implied_prob_pair"]

                        for meta_col in ("closing_bookmaker", "closing_line_time"):
                            pair_col = f"{meta_col}_pair"
                            if pair_col in working.columns:
                                mask = (
                                    working[meta_col].isna()
                                    & working[pair_col].notna()
                                )
                                working.loc[mask, meta_col] = working.loc[mask, pair_col]

                        drop_cols = ["home_team_pair", "away_team_pair"]
                        drop_cols.extend(f"{col}_pair" for col in available_cols)
                        working.drop(columns=drop_cols, inplace=True, errors="ignore")

                    working.drop(columns=["_team_pair"], inplace=True, errors="ignore")
                    supplemental.drop(columns=["_team_pair"], inplace=True, errors="ignore")


                if temporary_game_id_key is not None:
                    working.drop(columns=[temporary_game_id_key], inplace=True, errors="ignore")
                    supplemental.drop(columns=[temporary_game_id_key], inplace=True, errors="ignore")
                if "_merge_start_date" in working.columns:
                    working.drop(columns=["_merge_start_date"], inplace=True, errors="ignore")
                if "_merge_start_date" in supplemental.columns:
                    supplemental.drop(columns=["_merge_start_date"], inplace=True, errors="ignore")

        working.drop(columns=["_start_date"], inplace=True, errors="ignore")

        inferred_mask = pd.Series(False, index=working.index)
        for side in ("home", "away"):
            closing_col = f"{side}_closing_moneyline"
            fallback_col = f"{side}_moneyline"
            closing_prob_col = f"{side}_closing_implied_prob"

            if closing_col not in working.columns or fallback_col not in working.columns:
                continue

            missing = working[closing_col].isna() & working[fallback_col].notna()
            if not missing.any():
                continue

            fallback_probs = self._american_to_prob_series(working.loc[missing, fallback_col])
            if closing_prob_col in working.columns:
                combined = working.loc[missing, closing_prob_col]
                combined = combined.fillna(fallback_probs)
                working.loc[missing, closing_prob_col] = combined
            else:
                working.loc[missing, closing_prob_col] = fallback_probs

            inferred_mask = inferred_mask | missing

        if inferred_mask.any():
            inferred_count = int(inferred_mask.sum())
            logging.warning(
                "Closing odds are missing for %d games; they will be treated as unavailable until verified closing lines are loaded.",
                inferred_count,
            )
            closing_cols = [
                "home_closing_moneyline",
                "away_closing_moneyline",
                "home_closing_implied_prob",
                "away_closing_implied_prob",
                "closing_bookmaker",
                "closing_line_time",
            ]
            for col in closing_cols:
                if col in working.columns:
                    working.loc[inferred_mask, col] = np.nan

        return working

    def _build_team_game_lookup(self, games: pd.DataFrame) -> pd.DataFrame:
        if games is None or games.empty:
            return pd.DataFrame(columns=["team", "opponent", "is_home", "start_time", "game_id"])

        working = games.copy()
        working["start_time"] = pd.to_datetime(working["start_time"], utc=True, errors="coerce")

        home = working[["game_id", "start_time", "home_team", "away_team"]].rename(
            columns={"home_team": "team", "away_team": "opponent"}
        )
        home["is_home"] = True

        away = working[["game_id", "start_time", "away_team", "home_team"]].rename(
            columns={"away_team": "team", "home_team": "opponent"}
        )
        away["is_home"] = False

        combined = safe_concat([home, away], ignore_index=True)
        if combined.empty:
            return pd.DataFrame(columns=["team", "opponent", "is_home", "start_time", "game_id"])

        combined["team"] = combined["team"].apply(normalize_team_abbr)
        combined["opponent"] = combined["opponent"].apply(normalize_team_abbr)
        combined = combined.dropna(subset=["team", "opponent", "start_time"])
        combined = combined.sort_values(["team", "start_time", "game_id"]).reset_index(drop=True)
        return combined

    def _lookup_previous_game(
        self, team: Optional[str], reference_time: Optional[pd.Timestamp]
    ) -> Optional[pd.Series]:
        if not team:
            return None
        lookup = self.team_game_lookup
        if lookup is None or lookup.empty:
            return None

        try:
            ref = pd.to_datetime(reference_time, utc=True, errors="coerce")
        except Exception:
            ref = pd.NaT

        team_rows = lookup[lookup["team"] == normalize_team_abbr(team)]
        if team_rows.empty:
            return None

        if pd.notna(ref):
            team_rows = team_rows[team_rows["start_time"] < ref]
            if team_rows.empty:
                return None
        return team_rows.iloc[-1]

    @staticmethod
    def _tz_offset_hours(ts: Optional[pd.Timestamp], team: Optional[str]) -> float:
        if ts is None or pd.isna(ts):
            return 0.0
        if ts.tzinfo is None:
            ts = ts.tz_localize(dt.timezone.utc)
        tz_name = TEAM_TIMEZONES.get(team or "", "UTC")
        try:
            zone = ZoneInfo(tz_name)
        except Exception:
            zone = ZoneInfo("UTC")
        offset = ts.astimezone(zone).utcoffset()
        if offset is None:
            return 0.0
        return float(offset.total_seconds() / 3600.0)

    def _fallback_team_context(
        self,
        team: Optional[str],
        opponent: Optional[str],
        start_time: Optional[pd.Timestamp],
        *,
        is_home: bool,
    ) -> Dict[str, float]:
        defaults: Dict[str, float] = {
            "rest_days": np.nan,
            "rest_penalty": np.nan,
            "travel_penalty": np.nan,
            "timezone_diff_hours": np.nan,
            "avg_timezone_diff_hours": np.nan,
        }

        if not team:
            return defaults

        try:
            start_ts = pd.to_datetime(start_time, utc=True, errors="coerce")
        except Exception:
            start_ts = pd.NaT

        if pd.isna(start_ts):
            return defaults

        previous = self._lookup_previous_game(team, start_ts)
        if previous is not None:
            prev_start = pd.to_datetime(previous.get("start_time"), utc=True, errors="coerce")
            if pd.notna(prev_start):
                rest_days = float((start_ts - prev_start).total_seconds() / 86400.0)
                if rest_days >= 0:
                    defaults["rest_days"] = rest_days
                    defaults["rest_penalty"] = max(0.0, 6.0 - rest_days)

        venue_team = team if is_home else opponent
        tz_team = self._tz_offset_hours(start_ts, team)
        tz_venue = self._tz_offset_hours(start_ts, venue_team)
        tz_diff = abs(tz_team - tz_venue)
        defaults["timezone_diff_hours"] = tz_diff
        defaults["avg_timezone_diff_hours"] = tz_diff
        defaults["travel_penalty"] = 0.0 if is_home else tz_diff / 3.0
        return defaults

    def load_dataframes(
        self,
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
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
        depth_charts = pd.DataFrame()
        advanced_metrics = pd.read_sql_table("nfl_team_advanced_metrics", self.engine)
        try:
            game_totals = pd.read_sql_table("nfl_game_totals", self.engine)
        except Exception:
            game_totals = pd.DataFrame()
        try:
            player_prop_lines = pd.read_sql_table("nfl_player_prop_lines", self.engine)
        except Exception:
            player_prop_lines = pd.DataFrame()

        # Normalize column names to plain strings so downstream pipelines see
        # consistent labels regardless of database dialect.
        games = games.rename(columns=lambda col: str(col))
        player_stats = player_stats.rename(columns=lambda col: str(col))
        team_ratings = team_ratings.rename(columns=lambda col: str(col))
        injuries = injuries.rename(columns=lambda col: str(col))
        depth_charts = depth_charts.rename(columns=lambda col: str(col))
        advanced_metrics = advanced_metrics.rename(columns=lambda col: str(col))
        game_totals = game_totals.rename(columns=lambda col: str(col))
        player_prop_lines = player_prop_lines.rename(columns=lambda col: str(col))

        loader = getattr(self, "supplemental_loader", None)
        travel_context = None
        if loader is not None:
            travel_context = getattr(loader, "travel_context_frame", None)
        if travel_context is not None and not travel_context.empty:
            travel_df = travel_context.copy()
            travel_df["team"] = travel_df["team"].apply(normalize_team_abbr)
            travel_df["season"] = travel_df["season"].astype(str)
            if "week" in travel_df.columns:
                travel_df["week"] = travel_df["week"].apply(
                    lambda x: int(x) if pd.notna(x) else None
                )

            def _merge_travel(prefix: str, team_col: str) -> Optional[pd.DataFrame]:
                if not {"season", "week", team_col}.issubset(games.columns):
                    return None
                context_cols = [
                    "rest_days",
                    "rest_penalty",
                    "travel_penalty",
                    "timezone_diff_hours",
                    "avg_timezone_diff_hours",
                ]
                available = [col for col in context_cols if col in travel_df.columns]
                if not available:
                    return None
                rename_map = {col: f"{prefix}_{col}" for col in available}
                merge_frame = travel_df[["season", "week", "team", *available]].rename(
                    columns=rename_map | {"team": team_col}
                )
                key_cols = ["season", "week", team_col]
                merged = games.merge(merge_frame, on=key_cols, how="left")

                for col in available:
                    target_col = f"{prefix}_{col}"
                    if target_col in merged.columns and col in merged.columns:
                        merged[target_col] = merged[target_col].combine_first(merged[col])
                for col in available:
                    merged.drop(columns=[col], inplace=True, errors="ignore")
                return merged

            merged_home = _merge_travel("home", "home_team")
            if merged_home is not None:
                games = merged_home
            merged_away = _merge_travel("away", "away_team")
            if merged_away is not None:
                games = merged_away
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

        if "team" in player_prop_lines.columns:
            player_prop_lines["team"] = player_prop_lines["team"].apply(normalize_team_abbr)
        if "opponent" in player_prop_lines.columns:
            player_prop_lines["opponent"] = player_prop_lines["opponent"].apply(normalize_team_abbr)

        return (
            games,
            player_stats,
            team_ratings,
            injuries,
            depth_charts,
            advanced_metrics,
            game_totals,
            player_prop_lines,
        )

    def build_features(self) -> Dict[str, pd.DataFrame]:
        (
            games,
            player_stats,
            team_ratings,
            injuries,
            depth_charts,
            advanced_metrics,
            game_totals,
            player_prop_lines,
        ) = self.load_dataframes()

        if games.empty:
            logging.warning("No games available in the database. Skipping model training.")
            return {}

        games = games.copy()
        self.games_frame_raw = games.copy()
        games = self._backfill_game_odds(games)
        if "home_moneyline" in games.columns and "away_moneyline" in games.columns:
            odds_lookup = games.dropna(subset=["home_moneyline", "away_moneyline"], how="any").copy()
            if not odds_lookup.empty:
                odds_lookup["start_time"] = pd.to_datetime(odds_lookup["start_time"], errors="coerce")
                odds_lookup["_start_date"] = odds_lookup["start_time"].dt.normalize()
                sort_cols = [col for col in ["odds_updated", "start_time"] if col in odds_lookup.columns]
                if sort_cols:
                    odds_lookup = odds_lookup.sort_values(sort_cols, ascending=[False] * len(sort_cols))
                odds_lookup = odds_lookup.drop_duplicates(
                    subset=["home_team", "away_team", "_start_date"], keep="first"
                )
                self.latest_odds_lookup = odds_lookup[
                    [
                        "home_team",
                        "away_team",
                        "_start_date",
                        "home_moneyline",
                        "away_moneyline",
                        "home_implied_prob",
                        "away_implied_prob",
                        "odds_updated",
                    ]
                ].rename(columns={"_start_date": "start_date"})
            else:
                self.latest_odds_lookup = pd.DataFrame()
        else:
            self.latest_odds_lookup = pd.DataFrame()

        if not game_totals.empty and "game_id" in game_totals.columns:
            for col in ["last_update", "start_time"]:
                if col in game_totals.columns:
                    game_totals[col] = pd.to_datetime(game_totals[col], errors="coerce")
            sort_cols = [col for col in ["last_update", "start_time"] if col in game_totals.columns]
            if sort_cols:
                totals_sorted = game_totals.sort_values(sort_cols, ascending=[False] * len(sort_cols))
            else:
                totals_sorted = game_totals
            totals_latest = totals_sorted.drop_duplicates(subset=["game_id"], keep="first")
            rename_map = {
                "total_line": "game_total_line",
                "over_odds": "game_total_over_odds",
                "under_odds": "game_total_under_odds",
            }
            totals_latest = totals_latest[[col for col in rename_map if col in totals_latest.columns] + ["game_id"]]
            totals_latest = totals_latest.rename(columns=rename_map)
            games = games.merge(totals_latest, on="game_id", how="left")

        player_stats = player_stats.copy()
        injuries = injuries.copy()
        depth_charts = depth_charts.copy()
        advanced_metrics = advanced_metrics.copy()
        game_totals = game_totals.copy()
        player_prop_lines = player_prop_lines.copy()

        if "position" in player_stats.columns:
            player_stats["position"] = player_stats["position"].apply(normalize_position)
        self.games_frame = games
        self.injury_frame = injuries
        self.depth_chart_frame = depth_charts
        self.advanced_metrics_frame = advanced_metrics
        self.game_totals_frame = game_totals
        self.player_prop_lines_frame = player_prop_lines
        self.team_game_lookup = self._build_team_game_lookup(games)

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
        self.team_game_lookup = team_game_history
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

        def _ensure_penalty_columns(strength: pd.DataFrame) -> pd.DataFrame:
            if strength is None or strength.empty:
                return strength
            for penalty_col in ("travel_penalty", "rest_penalty", "avg_timezone_diff_hours"):
                if penalty_col not in strength.columns:
                    strength[penalty_col] = 0.0
                else:
                    strength[penalty_col] = pd.to_numeric(
                        strength[penalty_col], errors="coerce"
                    ).fillna(0.0)
            return strength

        if player_stats.empty:
            logging.warning(
                "Player statistics table is empty. Player-level models will not be trained."
            )
            team_strength = self._compute_team_unit_strength(player_stats, advanced_metrics)
            team_strength = _merge_penalties_into_strength(team_strength)
            team_strength = _ensure_penalty_columns(team_strength)
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
            team_strength = _ensure_penalty_columns(team_strength)

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
                    "offense_yards_per_play": "opp_offense_yards_per_play",
                    "defense_yards_per_play": "opp_defense_yards_per_play",
                    "offense_td_rate": "opp_offense_td_rate",
                    "defense_td_rate": "opp_defense_td_rate",
                    "pass_rate": "opp_pass_rate",
                    "rush_rate": "opp_rush_rate",
                    "pass_rate_over_expectation": "opp_pass_rate_over_expectation",
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

            if not player_prop_lines.empty:
                props = player_prop_lines.copy()
                props["game_id"] = props["game_id"].astype(str)
                if "player_name_norm" in props.columns:
                    props["player_name_norm"] = props["player_name_norm"].fillna("").astype(str)
                else:
                    props["player_name_norm"] = ""
                empty_norm = props["player_name_norm"].str.strip() == ""
                if empty_norm.any():
                    props.loc[empty_norm, "player_name_norm"] = props.loc[empty_norm, "player_name"].fillna("").map(
                        normalize_player_name
                    )
                props["player_name_norm"] = props["player_name_norm"].fillna("").map(normalize_player_name)
                props = props[props["player_name_norm"] != ""].copy()
                if "last_update" in props.columns:
                    props["last_update"] = pd.to_datetime(props["last_update"], errors="coerce")
                    props = props.sort_values("last_update", ascending=False)
                props = props.drop_duplicates(
                    subset=["game_id", "player_name_norm", "market"], keep="first"
                )
                for market_key, column_name in PLAYER_PROP_MARKET_COLUMN_MAP.items():
                    subset = props[props["market"] == market_key]
                    if subset.empty:
                        continue
                    stat_name = column_name.replace("line_", "")
                    over_col = f"over_odds_{stat_name}"
                    under_col = f"under_odds_{stat_name}"
                    subset_merge = subset[
                        ["game_id", "player_name_norm", "line", "over_odds", "under_odds"]
                    ].rename(
                        columns={
                            "line": column_name,
                            "over_odds": over_col,
                            "under_odds": under_col,
                        }
                    )
                    player_stats = player_stats.merge(
                        subset_merge,
                        on=["game_id", "player_name_norm"],
                        how="left",
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
                subset_all = player_stats[
                    player_stats["position"].isin(list(positions))
                ].copy()
                if subset_all.empty:
                    logging.debug(
                        "Skipping %s dataset because no positional rows are available", target
                    )
                    return

                subset_all["team"] = subset_all["team"].apply(normalize_team_abbr)
                subset_all["position"] = subset_all["position"].apply(normalize_position)
                subset_all["_usage_weight"] = compute_recency_usage_weights(subset_all)
                subset_all["_usage_weight"] = (
                    subset_all["_usage_weight"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
                )

                labeled = subset_all[subset_all[target].notna()].copy()
                if labeled.empty:
                    logging.debug(
                        "Skipping %s dataset because no labeled rows are available", target
                    )
                    return

                labeled["is_synthetic"] = False
                labeled["sample_weight"] = labeled["_usage_weight"].clip(lower=1e-4)

                combined = labeled.drop(columns=["_usage_weight"], errors="ignore")
                missing_count = int(subset_all[target].isna().sum())
                if missing_count:
                    logging.warning(
                        "%d %s rows are missing outcomes. Populate historical stats instead of relying on synthetic priors.",
                        missing_count,
                        target,
                    )
                datasets[target] = combined

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
                "offense_yards_per_play": "home_offense_yards_per_play",
                "defense_yards_per_play": "home_defense_yards_per_play",
                "offense_td_rate": "home_offense_td_rate",
                "defense_td_rate": "home_defense_td_rate",
                "pass_rate": "home_pass_rate",
                "rush_rate": "home_rush_rate",
                "pass_rate_over_expectation": "home_pass_rate_over_expectation",
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
                "offense_yards_per_play": "away_offense_yards_per_play",
                "defense_yards_per_play": "away_defense_yards_per_play",
                "offense_td_rate": "away_offense_td_rate",
                "defense_td_rate": "away_defense_td_rate",
                "pass_rate": "away_pass_rate",
                "rush_rate": "away_rush_rate",
                "pass_rate_over_expectation": "away_pass_rate_over_expectation",
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
                "team_elo_pre": "home_elo_pre",
                "team_elo_post": "home_elo_post",
                "team_elo_change": "home_elo_change",
                "team_elo_win_prob": "home_elo_win_prob",
                "team_elo_vs_opponent": "home_elo_vs_opponent",
                "opponent_elo_pre": "home_opponent_elo_pre",
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
                "team_elo_pre": "away_elo_pre",
                "team_elo_post": "away_elo_post",
                "team_elo_change": "away_elo_change",
                "team_elo_win_prob": "away_elo_win_prob",
                "team_elo_vs_opponent": "away_elo_vs_opponent",
                "opponent_elo_pre": "away_opponent_elo_pre",
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

        games_context = self._augment_matchup_features(games_context)

        if "home_travel_penalty" not in games_context.columns:
            games_context["home_travel_penalty"] = np.nan
        if "away_travel_penalty" not in games_context.columns:
            games_context["away_travel_penalty"] = np.nan

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

        penalty_fill_cols = [
            "home_travel_penalty",
            "home_rest_penalty",
            "home_timezone_diff_hours",
            "home_rest_days",
            "away_travel_penalty",
            "away_rest_penalty",
            "away_timezone_diff_hours",
            "away_rest_days",
        ]
        for col in penalty_fill_cols:
            if col not in games_context.columns:
                games_context[col] = 0.0
            else:
                games_context[col] = pd.to_numeric(
                    games_context[col], errors="coerce"
                ).fillna(0.0)

        elo_cols = [
            "home_elo_pre",
            "home_elo_post",
            "home_elo_change",
            "home_elo_win_prob",
            "home_elo_vs_opponent",
            "home_opponent_elo_pre",
            "home_elo_edge",
            "away_elo_pre",
            "away_elo_post",
            "away_elo_change",
            "away_elo_win_prob",
            "away_elo_vs_opponent",
            "away_opponent_elo_pre",
            "away_elo_edge",
        ]
        for col in elo_cols:
            if col in games_context.columns:
                games_context[col] = pd.to_numeric(
                    games_context[col], errors="coerce"
                )
            else:
                games_context[col] = np.nan

        games_context["moneyline_diff"] = games_context["home_moneyline"] - games_context["away_moneyline"]
        games_context["implied_prob_diff"] = (
            games_context["home_implied_prob"] - games_context["away_implied_prob"]
        )
        games_context["implied_prob_sum"] = (
            games_context["home_implied_prob"] + games_context["away_implied_prob"]
        )

        if {
            "home_elo_win_prob",
            "away_elo_win_prob",
            "home_implied_prob",
            "away_implied_prob",
        }.issubset(games_context.columns):
            games_context["home_elo_edge"] = (
                games_context["home_elo_win_prob"] - games_context["home_implied_prob"]
            )
            games_context["away_elo_edge"] = (
                games_context["away_elo_win_prob"] - games_context["away_implied_prob"]
            )
        else:
            games_context["home_elo_edge"] = np.nan
            games_context["away_elo_edge"] = np.nan

        games_context["point_diff"] = games_context["home_score"] - games_context["away_score"]
        self.games_frame = games_context.copy()
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

    def _get_latest_team_history(
        self,
        team: Optional[str],
        season: Optional[str],
        reference_time: Optional[pd.Timestamp] = None,
    ) -> Optional[pd.Series]:
        if not team:
            return None

        history = self.team_history_frame
        ref_ts: Optional[pd.Timestamp]
        try:
            ref_ts = pd.to_datetime(reference_time, utc=True, errors="coerce")
        except Exception:
            ref_ts = None

        if history is not None and not history.empty:
            subset = history[history["team"] == normalize_team_abbr(team)]
            if ref_ts is not None and not pd.isna(ref_ts):
                subset = subset[pd.to_datetime(subset["start_time"], utc=True, errors="coerce") < ref_ts]
            if season:
                season_subset = subset[subset["season"] == season]
                if not season_subset.empty:
                    return season_subset.sort_values("start_time").iloc[-1]
            if not subset.empty:
                return subset.sort_values("start_time").iloc[-1]

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

        fallback = self._fallback_team_context(team, None, ref_ts, is_home=True)
        if any(not pd.isna(value) for value in fallback.values()):
            return pd.Series(fallback)
        return None

    def prepare_upcoming_game_features(self, upcoming_games: pd.DataFrame) -> pd.DataFrame:
        if upcoming_games.empty:
            return upcoming_games.copy()

        features = upcoming_games.copy()
        features["start_time"] = pd.to_datetime(features["start_time"], utc=True, errors="coerce")
        if "day_of_week" not in features.columns or features["day_of_week"].isna().any():
            features["day_of_week"] = features["start_time"].dt.day_name()

        # Reuse the latest odds snapshot harvested during feature assembly so upcoming
        # games inherit backfilled closing numbers even when the source table is sparse.
        lookup = self.latest_odds_lookup
        if lookup is not None and not lookup.empty:
            odds_lookup = lookup.copy()
            odds_lookup["start_date"] = pd.to_datetime(
                odds_lookup["start_date"], errors="coerce"
            )
            features["_start_date"] = features["start_time"].dt.normalize()
            features = self._merge_odds_snapshot(
                features,
                odds_lookup,
                ["home_team", "away_team", "_start_date"],
                [
                    "home_moneyline",
                    "away_moneyline",
                    "home_implied_prob",
                    "away_implied_prob",
                    "odds_updated",
                ],
                sort_cols=["odds_updated"],
            )
            features.drop(columns=["_start_date"], inplace=True, errors="ignore")

        numeric_placeholders = {
            "home_moneyline": np.nan,
            "away_moneyline": np.nan,
            "home_implied_prob": np.nan,
            "away_implied_prob": np.nan,
            "home_offense_pass_rating": np.nan,
            "home_offense_rush_rating": np.nan,
            "home_defense_pass_rating": np.nan,
            "home_defense_rush_rating": np.nan,
            "home_pace_seconds_per_play": np.nan,
            "home_offense_epa": np.nan,
            "home_defense_epa": np.nan,
            "home_offense_success_rate": np.nan,
            "home_defense_success_rate": np.nan,
            "home_offense_yards_per_play": np.nan,
            "home_defense_yards_per_play": np.nan,
            "home_offense_td_rate": np.nan,
            "home_defense_td_rate": np.nan,
            "home_pass_rate": np.nan,
            "home_rush_rate": np.nan,
            "home_pass_rate_over_expectation": np.nan,
            "home_travel_penalty": np.nan,
            "home_rest_penalty": np.nan,
            "home_weather_adjustment": np.nan,
            "home_timezone_diff_hours": np.nan,
            "home_avg_timezone_diff_hours": np.nan,
            "away_offense_pass_rating": np.nan,
            "away_offense_rush_rating": np.nan,
            "away_defense_pass_rating": np.nan,
            "away_defense_rush_rating": np.nan,
            "away_pace_seconds_per_play": np.nan,
            "away_offense_epa": np.nan,
            "away_defense_epa": np.nan,
            "away_offense_success_rate": np.nan,
            "away_defense_success_rate": np.nan,
            "away_offense_yards_per_play": np.nan,
            "away_defense_yards_per_play": np.nan,
            "away_offense_td_rate": np.nan,
            "away_defense_td_rate": np.nan,
            "away_pass_rate": np.nan,
            "away_rush_rate": np.nan,
            "away_pass_rate_over_expectation": np.nan,
            "away_travel_penalty": np.nan,
            "away_rest_penalty": np.nan,
            "away_weather_adjustment": np.nan,
            "away_timezone_diff_hours": np.nan,
            "away_avg_timezone_diff_hours": np.nan,
            "home_points_for_avg": np.nan,
            "home_points_against_avg": np.nan,
            "home_point_diff_avg": np.nan,
            "home_win_pct_recent": np.nan,
            "home_prev_points_for": np.nan,
            "home_prev_points_against": np.nan,
            "home_prev_point_diff": np.nan,
            "home_rest_days": np.nan,
            "home_injury_total": 0.0,
            "home_elo_pre": np.nan,
            "home_elo_post": np.nan,
            "home_elo_change": np.nan,
            "home_elo_win_prob": np.nan,
            "home_elo_vs_opponent": np.nan,
            "home_opponent_elo_pre": np.nan,
            "home_elo_edge": np.nan,
            "away_points_for_avg": np.nan,
            "away_points_against_avg": np.nan,
            "away_point_diff_avg": np.nan,
            "away_win_pct_recent": np.nan,
            "away_prev_points_for": np.nan,
            "away_prev_points_against": np.nan,
            "away_prev_point_diff": np.nan,
            "away_rest_days": np.nan,
            "away_elo_pre": np.nan,
            "away_elo_post": np.nan,
            "away_elo_change": np.nan,
            "away_elo_win_prob": np.nan,
            "away_elo_vs_opponent": np.nan,
            "away_opponent_elo_pre": np.nan,
            "away_elo_edge": np.nan,
            "away_injury_total": 0.0,
            "offense_pass_rating_diff": 0.0,
            "offense_rush_rating_diff": 0.0,
            "defense_pass_rating_diff": 0.0,
            "defense_rush_rating_diff": 0.0,
            "offense_epa_diff": 0.0,
            "defense_epa_diff": 0.0,
            "offense_success_rate_diff": 0.0,
            "defense_success_rate_diff": 0.0,
            "offense_yards_per_play_diff": 0.0,
            "defense_yards_per_play_diff": 0.0,
            "offense_td_rate_diff": 0.0,
            "defense_td_rate_diff": 0.0,
            "pass_rate_diff": 0.0,
            "rush_rate_diff": 0.0,
            "pass_rate_over_expectation_diff": 0.0,
            "pace_seconds_diff": 0.0,
            "travel_penalty_diff": 0.0,
            "rest_penalty_diff": 0.0,
            "weather_adjustment_diff": 0.0,
            "timezone_diff_diff": 0.0,
            "points_for_avg_diff": 0.0,
            "points_against_avg_diff": 0.0,
            "point_diff_avg_diff": 0.0,
            "win_pct_recent_diff": 0.0,
            "rest_days_diff": 0.0,
            "prev_points_for_diff": 0.0,
            "prev_points_against_diff": 0.0,
            "prev_point_diff_diff": 0.0,
            "injury_total_diff": 0.0,
            "implied_prob_logit_diff": 0.0,
            "home_net_epa": 0.0,
            "away_net_epa": 0.0,
            "home_net_success_rate": 0.0,
            "away_net_success_rate": 0.0,
            "home_net_yards_per_play": 0.0,
            "away_net_yards_per_play": 0.0,
            "home_net_td_rate": 0.0,
            "away_net_td_rate": 0.0,
            "home_pass_matchup": 0.0,
            "away_pass_matchup": 0.0,
            "home_rush_matchup": 0.0,
            "away_rush_matchup": 0.0,
            "wind_mph": np.nan,
            "humidity": np.nan,
        }
        missing_numeric_cols = {
            col: default
            for col, default in numeric_placeholders.items()
            if col not in features.columns
        }
        if missing_numeric_cols:
            filler = pd.DataFrame(
                {col: default for col, default in missing_numeric_cols.items()},
                index=features.index,
            )
            features = pd.concat([features, filler], axis=1)

        loader = getattr(self, "supplemental_loader", None)
        travel_context = None
        if loader is not None:
            travel_context = getattr(loader, "travel_context_frame", None)
        if travel_context is not None and not travel_context.empty:
            travel_df = travel_context.copy()
            travel_df["team"] = travel_df["team"].apply(normalize_team_abbr)
            travel_df["season"] = travel_df["season"].astype(str)
            if "week" in travel_df.columns:
                travel_df["week"] = travel_df["week"].apply(
                    lambda x: int(x) if pd.notna(x) else None
                )

            def _merge_upcoming(prefix: str, team_col: str) -> None:
                if not {"season", "week", team_col}.issubset(features.columns):
                    return
                context_cols = [
                    "rest_days",
                    "rest_penalty",
                    "travel_penalty",
                    "timezone_diff_hours",
                    "avg_timezone_diff_hours",
                ]
                available = [col for col in context_cols if col in travel_df.columns]
                if not available:
                    return
                rename_map = {col: f"{prefix}_{col}_supp" for col in available}
                merge_frame = travel_df[["season", "week", "team", *available]].rename(
                    columns=rename_map | {"team": team_col}
                )
                merged = features.merge(
                    merge_frame,
                    on=["season", "week", team_col],
                    how="left",
                )
                for col in available:
                    target_col = f"{prefix}_{col}"
                    supp_col = f"{prefix}_{col}_supp"
                    if supp_col not in merged.columns:
                        continue
                    if target_col in merged.columns:
                        merged[target_col] = merged[target_col].combine_first(merged[supp_col])
                    else:
                        merged[target_col] = merged[supp_col]
                    merged.drop(columns=[supp_col], inplace=True)
                return merged

            merged_home = _merge_upcoming("home", "home_team")
            if merged_home is not None:
                features = merged_home
            merged_away = _merge_upcoming("away", "away_team")
            if merged_away is not None:
                features = merged_away

        for idx, row in features.iterrows():
            season = row.get("season")
            home_team = row.get("home_team")
            away_team = row.get("away_team")
            start_time = row.get("start_time")

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
                    features.at[idx, "home_offense_yards_per_play"] = strength.get(
                        "offense_yards_per_play"
                    )
                    features.at[idx, "home_defense_yards_per_play"] = strength.get(
                        "defense_yards_per_play"
                    )
                    features.at[idx, "home_offense_td_rate"] = strength.get("offense_td_rate")
                    features.at[idx, "home_defense_td_rate"] = strength.get("defense_td_rate")
                    features.at[idx, "home_pass_rate"] = strength.get("pass_rate")
                    features.at[idx, "home_rush_rate"] = strength.get("rush_rate")
                    features.at[idx, "home_pass_rate_over_expectation"] = strength.get(
                        "pass_rate_over_expectation"
                    )
                    features.at[idx, "home_travel_penalty"] = strength.get("travel_penalty")
                    features.at[idx, "home_rest_penalty"] = strength.get("rest_penalty")
                    features.at[idx, "home_weather_adjustment"] = strength.get("weather_adjustment")
                    features.at[idx, "home_timezone_diff_hours"] = strength.get("avg_timezone_diff_hours")
                history = self._get_latest_team_history(home_team, season, start_time)
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
                    latest_elo = history.get("team_elo_post")
                    if pd.isna(latest_elo):
                        latest_elo = history.get("team_elo_pre")
                    features.at[idx, "home_elo_pre"] = latest_elo
                    features.at[idx, "home_elo_post"] = latest_elo
                    features.at[idx, "home_elo_change"] = history.get("team_elo_change")

                fallback_home = self._fallback_team_context(
                    home_team,
                    away_team,
                    start_time,
                    is_home=True,
                )
                for key, value in fallback_home.items():
                    if pd.isna(value):
                        continue
                    if key == "avg_timezone_diff_hours":
                        target_col = "home_timezone_diff_hours"
                    else:
                        target_col = f"home_{key}"
                    if target_col in features.columns and pd.isna(features.at[idx, target_col]):
                        features.at[idx, target_col] = value

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
                    features.at[idx, "away_offense_yards_per_play"] = strength.get(
                        "offense_yards_per_play"
                    )
                    features.at[idx, "away_defense_yards_per_play"] = strength.get(
                        "defense_yards_per_play"
                    )
                    features.at[idx, "away_offense_td_rate"] = strength.get("offense_td_rate")
                    features.at[idx, "away_defense_td_rate"] = strength.get("defense_td_rate")
                    features.at[idx, "away_pass_rate"] = strength.get("pass_rate")
                    features.at[idx, "away_rush_rate"] = strength.get("rush_rate")
                    features.at[idx, "away_pass_rate_over_expectation"] = strength.get(
                        "pass_rate_over_expectation"
                    )
                    features.at[idx, "away_travel_penalty"] = strength.get("travel_penalty")
                    features.at[idx, "away_rest_penalty"] = strength.get("rest_penalty")
                    features.at[idx, "away_weather_adjustment"] = strength.get("weather_adjustment")
                    features.at[idx, "away_timezone_diff_hours"] = strength.get("avg_timezone_diff_hours")
                history = self._get_latest_team_history(away_team, season, start_time)
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
                    latest_elo = history.get("team_elo_post")
                    if pd.isna(latest_elo):
                        latest_elo = history.get("team_elo_pre")
                    features.at[idx, "away_elo_pre"] = latest_elo
                    features.at[idx, "away_elo_post"] = latest_elo
                    features.at[idx, "away_elo_change"] = history.get("team_elo_change")

                fallback_away = self._fallback_team_context(
                    away_team,
                    home_team,
                    start_time,
                    is_home=False,
                )
                for key, value in fallback_away.items():
                    if pd.isna(value):
                        continue
                    if key == "avg_timezone_diff_hours":
                        target_col = "away_timezone_diff_hours"
                    else:
                        target_col = f"away_{key}"
                    if target_col in features.columns and pd.isna(features.at[idx, target_col]):
                        features.at[idx, target_col] = value

        def _elo_expected(home_rating: float, away_rating: float) -> float:
            return 1.0 / (1.0 + 10 ** ((away_rating - (home_rating + 55.0)) / 400.0))

        if {"home_elo_pre", "away_elo_pre"}.issubset(features.columns):
            mask = features["home_elo_pre"].notna() & features["away_elo_pre"].notna()
            if mask.any():
                home_ratings = features.loc[mask, "home_elo_pre"].astype(float)
                away_ratings = features.loc[mask, "away_elo_pre"].astype(float)
                expected = home_ratings.combine(away_ratings, _elo_expected)
                features.loc[mask, "home_elo_win_prob"] = expected
                features.loc[mask, "away_elo_win_prob"] = 1.0 - expected
                features.loc[mask, "home_elo_vs_opponent"] = home_ratings - away_ratings
                features.loc[mask, "away_elo_vs_opponent"] = away_ratings - home_ratings
                features.loc[mask, "home_opponent_elo_pre"] = away_ratings
                features.loc[mask, "away_opponent_elo_pre"] = home_ratings
                if "home_implied_prob" in features.columns:
                    features.loc[mask, "home_elo_edge"] = (
                        features.loc[mask, "home_elo_win_prob"]
                        - features.loc[mask, "home_implied_prob"].astype(float)
                    )
                if "away_implied_prob" in features.columns:
                    features.loc[mask, "away_elo_edge"] = (
                        features.loc[mask, "away_elo_win_prob"]
                        - features.loc[mask, "away_implied_prob"].astype(float)
                    )

        features = self._augment_matchup_features(features)

        fill_defaults = {col: 0.0 for col in numeric_placeholders.keys()}
        features[list(fill_defaults.keys())] = features[list(fill_defaults.keys())].fillna(
            fill_defaults
        )

        derived_numeric_cols = {
            "moneyline_diff": features["home_moneyline"] - features["away_moneyline"],
            "implied_prob_diff": features["home_implied_prob"]
            - features["away_implied_prob"],
            "implied_prob_sum": features["home_implied_prob"]
            + features["away_implied_prob"],
        }
        features = features.assign(**derived_numeric_cols)

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

        # --- Begin: require real production history (career/preseason/college allowed) ---
        base_players = self.player_feature_frame  # includes historical per-game rows
        if (
            base_players is None
            or "player_id" not in base_players.columns
            or "game_id" not in base_players.columns
        ):
            hist_counts = pd.DataFrame({"player_id": [], "hist_game_count": []})
        else:
            hist_counts = (
                base_players.groupby("player_id")["game_id"]
                .nunique(dropna=True)
                .rename("hist_game_count")
                .reset_index()
            )

        latest_players = latest_players.merge(hist_counts, on="player_id", how="left")
        latest_players["hist_game_count"] = latest_players["hist_game_count"].fillna(0).astype(int)

        # Sum any season_* totals we just merged earlier; benign if absent
        season_total_cols = [c for c in latest_players.columns if c.startswith("season_")]
        if season_total_cols:
            latest_players["_season_total_sum"] = latest_players[season_total_cols].sum(axis=1, numeric_only=True)
        else:
            latest_players["_season_total_sum"] = 0.0

        # Optional: include preseason / college if you add those tables later
        def _safe_read(name: str) -> pd.DataFrame:
            try:
                return pd.read_sql_table(name, self.engine).rename(columns=lambda c: str(c))
            except Exception:
                return pd.DataFrame()

        pre_cols = _safe_read("nfl_preseason_stats")
        col_cols = _safe_read("nfl_college_stats")

        def _count_hist(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty or "player_id" not in df.columns:
                return pd.DataFrame({"player_id": [], "extra_hist_game_count": []})
            return (
                df.groupby("player_id")["game_id"].nunique(dropna=True)
                .rename("extra_hist_game_count")
                .reset_index()
            )

        extra_hist = []
        for df in (pre_cols, col_cols):
            part = _count_hist(df)
            if not part.empty:
                extra_hist.append(part)
        if extra_hist:
            extra_hist = pd.concat(extra_hist, ignore_index=True)
            extra_hist = extra_hist.groupby("player_id")["extra_hist_game_count"].sum().reset_index()
            latest_players = latest_players.merge(extra_hist, on="player_id", how="left")
            latest_players["extra_hist_game_count"] = latest_players["extra_hist_game_count"].fillna(0).astype(int)
        else:
            latest_players["extra_hist_game_count"] = 0

        # Final production gate:
        latest_players = latest_players[
            (latest_players["hist_game_count"] > 0)
            | (latest_players["extra_hist_game_count"] > 0)
            | (latest_players["_season_total_sum"] > 0)
        ].copy()

        # Clean up helper columns
        latest_players.drop(columns=["_season_total_sum"], inplace=True, errors="ignore")
        # --- End: require real production history ---

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
                        row_copy["offense_yards_per_play"] = strength.get("offense_yards_per_play")
                        row_copy["defense_yards_per_play"] = strength.get("defense_yards_per_play")
                        row_copy["offense_td_rate"] = strength.get("offense_td_rate")
                        row_copy["defense_td_rate"] = strength.get("defense_td_rate")
                        row_copy["pass_rate"] = strength.get("pass_rate")
                        row_copy["rush_rate"] = strength.get("rush_rate")
                        row_copy["pass_rate_over_expectation"] = strength.get(
                            "pass_rate_over_expectation"
                        )
                        row_copy["travel_penalty"] = strength.get("travel_penalty")
                        row_copy["rest_penalty"] = strength.get("rest_penalty")
                        row_copy["weather_adjustment"] = strength.get("weather_adjustment")
                        row_copy["avg_timezone_diff_hours"] = strength.get("avg_timezone_diff_hours")

                    history = self._get_latest_team_history(team, season, start_time)
                    if history is not None:
                        if pd.isna(row_copy.get("rest_penalty")):
                            row_copy["rest_penalty"] = history.get("rest_penalty")
                        if pd.isna(row_copy.get("travel_penalty")):
                            row_copy["travel_penalty"] = history.get("travel_penalty")
                        if pd.isna(row_copy.get("avg_timezone_diff_hours")):
                            row_copy["avg_timezone_diff_hours"] = history.get("timezone_diff_hours")

                    fallback_team = self._fallback_team_context(
                        team,
                        opponent,
                        start_time,
                        is_home=(team == home_team),
                    )
                    for key, value in fallback_team.items():
                        if pd.isna(value):
                            continue
                        target_key = key if key != "avg_timezone_diff_hours" else "avg_timezone_diff_hours"
                        if target_key in row_copy and pd.isna(row_copy.get(target_key)):
                            row_copy[target_key] = value

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
                        row_copy["opp_offense_yards_per_play"] = opp_strength.get(
                            "offense_yards_per_play"
                        )
                        row_copy["opp_defense_yards_per_play"] = opp_strength.get(
                            "defense_yards_per_play"
                        )
                        row_copy["opp_offense_td_rate"] = opp_strength.get("offense_td_rate")
                        row_copy["opp_defense_td_rate"] = opp_strength.get("defense_td_rate")
                        row_copy["opp_pass_rate"] = opp_strength.get("pass_rate")
                        row_copy["opp_rush_rate"] = opp_strength.get("rush_rate")
                        row_copy["opp_pass_rate_over_expectation"] = opp_strength.get(
                            "pass_rate_over_expectation"
                        )
                        row_copy["opp_travel_penalty"] = opp_strength.get("travel_penalty")
                        row_copy["opp_rest_penalty"] = opp_strength.get("rest_penalty")
                        row_copy["opp_weather_adjustment"] = opp_strength.get("weather_adjustment")
                        row_copy["opp_timezone_diff_hours"] = opp_strength.get("avg_timezone_diff_hours")

                    opp_history = self._get_latest_team_history(opponent, season, start_time)
                    if opp_history is not None:
                        if pd.isna(row_copy.get("opp_rest_penalty")):
                            row_copy["opp_rest_penalty"] = opp_history.get("rest_penalty")
                        if pd.isna(row_copy.get("opp_travel_penalty")):
                            row_copy["opp_travel_penalty"] = opp_history.get("travel_penalty")
                        if pd.isna(row_copy.get("opp_timezone_diff_hours")):
                            row_copy["opp_timezone_diff_hours"] = opp_history.get("timezone_diff_hours")

                    opp_fallback = self._fallback_team_context(
                        opponent,
                        team,
                        start_time,
                        is_home=(opponent == home_team),
                    )
                    for key, value in opp_fallback.items():
                        if pd.isna(value):
                            continue
                        target_key = (
                            f"opp_{key}"
                            if key != "avg_timezone_diff_hours"
                            else "opp_timezone_diff_hours"
                        )
                        if target_key in row_copy and pd.isna(row_copy.get(target_key)):
                            row_copy[target_key] = value

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

        penalty_cols = [
            "travel_penalty",
            "rest_penalty",
            "avg_timezone_diff_hours",
            "opp_travel_penalty",
            "opp_rest_penalty",
            "opp_timezone_diff_hours",
        ]
        for col in penalty_cols:
            if col not in player_features.columns:
                player_features[col] = 0.0
            else:
                player_features[col] = pd.to_numeric(
                    player_features[col], errors="coerce"
                ).fillna(0.0)

        player_features = player_features.drop(columns=["player_name_norm"], errors="ignore")

        return player_features

    @staticmethod
    def _compute_team_unit_strength(
        player_stats: pd.DataFrame, advanced_metrics: Optional[pd.DataFrame] = None
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
            "offense_yards_per_play",
            "defense_yards_per_play",
            "offense_td_rate",
            "defense_td_rate",
            "pass_rate",
            "rush_rate",
            "pass_rate_over_expectation",
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
                opp_rushing_tds=pd.NamedAgg(column="rushing_tds", aggfunc="sum"),
                opp_passing_tds=pd.NamedAgg(column="passing_tds", aggfunc="sum"),
            )
            .rename(columns={"opponent": "team"})
        )

        merged = offense.merge(defense, on=["season", "week", "team"], how="left")
        for td_col in ("opp_rushing_tds", "opp_passing_tds"):
            if td_col not in merged.columns:
                merged[td_col] = np.nan
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
        merged["offense_yards_per_play"] = merged["yards_per_play"]
        merged["pass_rate"] = np.where(
            merged["plays"] > 0,
            merged["passing_attempts"] / merged["plays"],
            np.nan,
        )
        merged["rush_rate"] = np.where(
            merged["plays"] > 0,
            merged["rushing_attempts"] / merged["plays"],
            np.nan,
        )
        merged["offense_td_rate"] = np.where(
            merged["plays"] > 0,
            (merged["rushing_tds"] + merged["passing_tds"]) / merged["plays"],
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
        merged["defense_yards_per_play"] = np.where(
            (merged["opp_rushing_attempts"] + merged["opp_passing_attempts"]) > 0,
            (merged["opp_rushing_yards"] + merged["opp_passing_yards"]) /
            (merged["opp_rushing_attempts"] + merged["opp_passing_attempts"]),
            np.nan,
        )
        total_allowed_tds = (
            merged["opp_rushing_tds"].fillna(0.0) + merged["opp_passing_tds"].fillna(0.0)
        )
        merged["defense_td_rate"] = np.where(
            (merged["opp_rushing_attempts"] + merged["opp_passing_attempts"]) > 0,
            total_allowed_tds
            / (merged["opp_rushing_attempts"] + merged["opp_passing_attempts"]),
            np.nan,
        )

        league = (
            merged.groupby(["season", "week"], as_index=False)[
                [
                    "rush_per_attempt",
                    "pass_per_attempt",
                    "allowed_rush_per_attempt",
                    "allowed_pass_per_attempt",
                    "yards_per_play",
                    "defense_yards_per_play",
                    "pass_rate",
                    "rush_rate",
                    "offense_td_rate",
                    "defense_td_rate",
                ]
            ]
            .mean()
            .rename(
                columns={
                    "rush_per_attempt": "league_rush_per_attempt",
                    "pass_per_attempt": "league_pass_per_attempt",
                    "allowed_rush_per_attempt": "league_allowed_rush_per_attempt",
                    "allowed_pass_per_attempt": "league_allowed_pass_per_attempt",
                    "yards_per_play": "league_yards_per_play",
                    "defense_yards_per_play": "league_defense_yards_per_play",
                    "pass_rate": "league_pass_rate",
                    "rush_rate": "league_rush_rate",
                    "offense_td_rate": "league_offense_td_rate",
                    "defense_td_rate": "league_defense_td_rate",
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
        merged["pass_rate_over_expectation"] = merged["pass_rate"] - merged["league_pass_rate"]

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
            "offense_yards_per_play",
            "defense_yards_per_play",
            "offense_td_rate",
            "defense_td_rate",
            "pass_rate",
            "rush_rate",
            "pass_rate_over_expectation",
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
                "offense_yards_per_play",
                "defense_yards_per_play",
                "offense_td_rate",
                "defense_td_rate",
                "pass_rate",
                "rush_rate",
                "pass_rate_over_expectation",
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
                    "offense_yards_per_play",
                    "defense_yards_per_play",
                    "offense_td_rate",
                    "defense_td_rate",
                    "pass_rate",
                    "rush_rate",
                    "pass_rate_over_expectation",
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

    @staticmethod
    def _compute_game_elo_history(games: pd.DataFrame) -> pd.DataFrame:
        if games.empty:
            return pd.DataFrame(
                columns=[
                    "game_id",
                    "team",
                    "opponent",
                    "team_elo_pre",
                    "team_elo_post",
                    "team_elo_change",
                    "team_elo_win_prob",
                    "team_elo_vs_opponent",
                    "opponent_elo_pre",
                ]
            )

        ratings: Dict[str, float] = defaultdict(lambda: 1500.0)
        home_field_advantage = 55.0
        k_factor = 20.0
        records: List[Dict[str, Any]] = []

        working = games.copy()
        working["start_time"] = pd.to_datetime(working["start_time"], utc=True, errors="coerce")
        working = working.sort_values("start_time")

        def _expected(r_a: float, r_b: float) -> float:
            return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

        def _margin_multiplier(point_diff: float, rating_gap: float) -> float:
            diff_abs = abs(point_diff)
            if diff_abs <= 0:
                return 1.0
            scaled_gap = max(0.0, abs(rating_gap))
            return math.log(diff_abs + 1.0) * (2.2 / ((scaled_gap * 0.001) + 2.2))

        for row in working.itertuples(index=False):
            home_team = normalize_team_abbr(getattr(row, "home_team", None))
            away_team = normalize_team_abbr(getattr(row, "away_team", None))
            if not home_team or not away_team:
                continue

            home_rating = ratings[home_team]
            away_rating = ratings[away_team]

            expected_home = _expected(home_rating + home_field_advantage, away_rating)
            expected_away = 1.0 - expected_home

            home_post = home_rating
            away_post = away_rating

            home_score = getattr(row, "home_score", np.nan)
            away_score = getattr(row, "away_score", np.nan)
            if pd.notna(home_score) and pd.notna(away_score):
                point_diff = float(home_score) - float(away_score)
                result_home = 1.0 if point_diff > 0 else (0.0 if point_diff < 0 else 0.5)
                result_away = 1.0 - result_home
                margin_mult = _margin_multiplier(point_diff, home_rating - away_rating)
                home_post = home_rating + k_factor * margin_mult * (result_home - expected_home)
                away_post = away_rating + k_factor * margin_mult * (result_away - expected_away)
                ratings[home_team] = home_post
                ratings[away_team] = away_post

            records.append(
                {
                    "game_id": getattr(row, "game_id", None),
                    "team": home_team,
                    "opponent": away_team,
                    "team_elo_pre": home_rating,
                    "team_elo_post": home_post,
                    "team_elo_win_prob": expected_home,
                    "opponent_elo_pre": away_rating,
                }
            )
            records.append(
                {
                    "game_id": getattr(row, "game_id", None),
                    "team": away_team,
                    "opponent": home_team,
                    "team_elo_pre": away_rating,
                    "team_elo_post": away_post,
                    "team_elo_win_prob": expected_away,
                    "opponent_elo_pre": home_rating,
                }
            )

        elo_df = pd.DataFrame.from_records(records)
        if elo_df.empty:
            return elo_df

        elo_df["team_elo_change"] = elo_df["team_elo_post"] - elo_df["team_elo_pre"]
        elo_df["team_elo_vs_opponent"] = elo_df["team_elo_pre"] - elo_df["opponent_elo_pre"]
        return elo_df

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
            "team_elo_pre",
            "team_elo_post",
            "team_elo_change",
            "team_elo_win_prob",
            "team_elo_vs_opponent",
            "opponent_elo_pre",
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

        elo_history = self._compute_game_elo_history(games)
        if elo_history.empty:
            for col in [
                "team_elo_pre",
                "team_elo_post",
                "team_elo_change",
                "team_elo_win_prob",
                "team_elo_vs_opponent",
                "opponent_elo_pre",
            ]:
                team_games[col] = np.nan
        else:
            elo_history["team"] = elo_history["team"].apply(normalize_team_abbr)
            elo_history["opponent"] = elo_history["opponent"].apply(normalize_team_abbr)
            team_games = team_games.merge(
                elo_history,
                on=["game_id", "team", "opponent"],
                how="left",
            )

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
    def __init__(
        self,
        engine: Engine,
        db: NFLDatabase,
        supplemental_loader: Optional[SupplementalDataLoader] = None,
        run_id: Optional[str] = None,
    ):
        self.engine = engine
        self.db = db
        self.feature_builder = FeatureBuilder(engine, supplemental_loader)
        self.run_id = run_id or uuid.uuid4().hex
        self.model_uncertainty: Dict[str, Dict[str, float]] = {}
        self.target_priors: Dict[str, Dict[str, Any]] = {}
        self.prior_engines: Dict[str, Optional[Dict[str, Any]]] = {}
        self.special_models: Dict[str, Dict[str, Any]] = {}
        self.feature_column_map: Dict[str, List[str]] = {}
        self.supplemental_loader = supplemental_loader

    @staticmethod
    def _build_confidence_profile(
        margins: np.ndarray, correct_mask: np.ndarray, bucket_quantiles: Sequence[float] = (0.33, 0.66)
    ) -> Optional[Dict[str, Any]]:
        """Summarize how accuracy varies with probability margins."""

        if margins.size == 0 or correct_mask.size == 0:
            return None

        valid = np.isfinite(margins) & np.isfinite(correct_mask)
        if not valid.any():
            return None

        margins = margins[valid]
        correct_mask = correct_mask[valid]
        correct_mask = correct_mask.astype(int)

        try:
            quantiles = np.quantile(margins, bucket_quantiles)
        except Exception:
            quantiles = []

        quantiles = [float(q) for q in quantiles if math.isfinite(q)]
        if not quantiles:
            quantiles = [0.1, 0.2]

        thresholds = [0.0]
        for value in quantiles:
            if value > thresholds[-1]:
                thresholds.append(value)
        thresholds.append(float("inf"))

        # Ensure strictly increasing cutpoints
        for idx in range(1, len(thresholds)):
            if thresholds[idx] <= thresholds[idx - 1]:
                thresholds[idx] = thresholds[idx - 1] + 1e-6

        bucket_labels = ["low", "medium", "high"][: len(thresholds) - 1]
        buckets: List[Dict[str, Any]] = []
        for label, lower, upper in zip(bucket_labels, thresholds[:-1], thresholds[1:]):
            mask = (margins >= lower) & (margins < upper)
            count = int(mask.sum())
            accuracy = float(correct_mask[mask].mean()) if count > 0 else float("nan")
            buckets.append(
                {
                    "label": label,
                    "lower": float(lower),
                    "upper": float(upper),
                    "accuracy": accuracy,
                    "count": count,
                }
            )

        preferred_label = bucket_labels[-1] if bucket_labels else None
        margin_threshold = thresholds[-2] if len(thresholds) >= 2 else 0.2

        edge_threshold = max(0.03, float(margin_threshold) / 2.0)

        return {
            "thresholds": thresholds[: len(bucket_labels) + 1],
            "buckets": buckets,
            "preferred_label": preferred_label,
            "margin_threshold": float(margin_threshold),
            "edge_threshold": float(edge_threshold),
        }

    @staticmethod
    def _build_error_profile(errors: np.ndarray) -> Optional[Dict[str, float]]:
        if errors.size == 0:
            return None

        errors = errors[np.isfinite(errors)]
        if errors.size == 0:
            return None

        quantile_levels = [0.5, 0.8, 0.9, 0.95]
        profile = {
            "mean": float(np.mean(errors)),
            "median": float(np.median(errors)),
        }
        for level in quantile_levels:
            try:
                profile[f"q{int(level * 100)}"] = float(np.quantile(errors, level))
            except Exception:
                profile[f"q{int(level * 100)}"] = float("nan")
        return profile

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
        lineup_roster_full = pd.DataFrame()

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
                    allowed_lineup_keys_all = {
                        (str(gid), team, name, pos)
                        for gid, team, name, pos in zip(
                            lineup_roster["game_id"],
                            lineup_roster["team"],
                            lineup_roster["__pname_key"],
                            lineup_roster["position"],
                        )
                        if name
                    }
                    allowed_lineup_keys_starters = {
                        key
                        for key, starter in zip(
                            zip(
                                lineup_roster["game_id"],
                                lineup_roster["team"],
                                lineup_roster["__pname_key"],
                                lineup_roster["position"],
                            ),
                            lineup_roster["is_starter"],
                        )
                        if starter == 1 and key[2]
                    }
                    allowed_lineup_keys_starters = {
                        (str(gid), team, name, pos)
                        for gid, team, name, pos in allowed_lineup_keys_starters
                    }
                else:
                    allowed_lineup_keys_all = {
                        (str(gid), team, name, pos)
                        for gid, team, name, pos in zip(
                            lineup_roster["game_id"],
                            lineup_roster["team"],
                            lineup_roster["__pname_key"],
                            lineup_roster["position"],
                        )
                        if name
                    }
                    allowed_lineup_keys_starters = allowed_lineup_keys_all.copy()
                lineup_roster_full = lineup_roster.copy()
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

        if not roster_frames:
            logging.info(
                "No lineup rows supplied for %d games; leaving player pool unchanged",
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

        if not lineup_roster_full.empty:
            overrides = lineup_roster_full[
                ["game_id", "team", "player_id", "__pname_key", "position"]
            ].copy()
            overrides["game_id"] = overrides["game_id"].astype(str)
            overrides["team"] = overrides["team"].apply(normalize_team_abbr)
            overrides["player_id"] = overrides["player_id"].fillna("").astype(str)
            overrides["__pname_key"] = overrides["__pname_key"].fillna("")
            overrides["position"] = overrides["position"].apply(normalize_position)

            pid_override: Dict[Tuple[str, str], Tuple[str, str]] = {}
            name_override: Dict[Tuple[str, str], Tuple[str, str]] = {}

            for _, row in overrides.iterrows():
                team_pos = (row["team"], row["position"])

                player_id_value = row.get("player_id", "")
                if isinstance(player_id_value, str):
                    player_id_value = player_id_value.strip()
                if player_id_value:
                    pid_override[(row["game_id"], player_id_value)] = team_pos

                name_key_value = row.get("__pname_key", "")
                if isinstance(name_key_value, str):
                    name_key_value = name_key_value.strip()
                if name_key_value:
                    name_override[(row["game_id"], name_key_value)] = team_pos

            player_df["_gid"] = player_df["game_id"].astype(str)
            player_df["_pid"] = player_df.get("player_id", "").fillna("").astype(str)

            team_overrides: List[Optional[str]] = []
            pos_overrides: List[Optional[str]] = []

            name_keys = player_df["__pname_key"].fillna("")
            for gid, pid, name_key in zip(player_df["_gid"], player_df["_pid"], name_keys):
                override = pid_override.get((gid, pid)) if pid else None
                if override is None and name_key:
                    override = name_override.get((gid, name_key))
                if override is None:
                    team_overrides.append(None)
                    pos_overrides.append(None)
                else:
                    team_overrides.append(override[0])
                    pos_overrides.append(override[1])

            team_overrides_series = pd.Series(team_overrides, index=player_df.index)
            pos_overrides_series = pd.Series(pos_overrides, index=player_df.index)

            team_mask = team_overrides_series.notna()
            if team_mask.any():
                player_df.loc[team_mask, "team"] = team_overrides_series[team_mask]

            pos_mask = pos_overrides_series.notna()
            if pos_mask.any():
                player_df.loc[pos_mask, "position"] = pos_overrides_series[pos_mask]

            player_df.drop(columns=["_gid", "_pid"], inplace=True)

        player_df["team"] = player_df["team"].apply(normalize_team_abbr)
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
        ).copy()

        merged["game_id"] = merged["game_id"].astype(str)
        merged["team"] = merged["team"].apply(normalize_team_abbr)
        merged["position"] = merged["position"].apply(normalize_position)
        if "is_placeholder" not in merged.columns:
            merged["is_placeholder"] = False
        else:
            merged["is_placeholder"] = merged["is_placeholder"].fillna(False)

        numeric_columns: List[str] = [
            col
            for col in merged.columns
            if pd.api.types.is_numeric_dtype(merged[col])
            and col not in {"depth_rank", "is_starter", "_lineup_hit"}
        ]

        merged["_placeholder_weight"] = compute_recency_usage_weights(merged)

        def _compute_weighted_baseline(
            frame: pd.DataFrame, group_cols: List[str]
        ) -> pd.DataFrame:
            if frame.empty or not numeric_columns:
                return pd.DataFrame()

            def _agg(group: pd.DataFrame) -> pd.Series:
                weights = group["_placeholder_weight"].fillna(0.0)
                positive_weight = float(weights[weights > 0].sum())
                result: Dict[str, float] = {}
                for column in numeric_columns:
                    values = group[column]
                    mask = values.notna()
                    if mask.any():
                        use_weights = weights[mask]
                        if use_weights.sum() > 0:
                            result[column] = float(
                                np.average(values[mask], weights=use_weights)
                            )
                        else:
                            result[column] = float(values[mask].mean())
                    else:
                        result[column] = np.nan
                if positive_weight <= 0:
                    positive_weight = float(len(group))
                result["_weight"] = positive_weight
                return pd.Series(result)

            baseline = (
                frame.groupby(group_cols, dropna=False, group_keys=False)
                .apply(_agg, include_groups=False)
                .sort_index()
            )
            baseline.index = baseline.index.set_names(group_cols)
            return baseline

        if numeric_columns:
            game_team_pos_baseline = _compute_weighted_baseline(
                merged, ["game_id", "team", "position"]
            )
            team_context_baseline = _compute_weighted_baseline(merged, ["game_id", "team"])
            team_pos_baseline = _compute_weighted_baseline(merged, ["team", "position"])
            pos_baseline = _compute_weighted_baseline(merged, ["position"])

            weights_all = merged["_placeholder_weight"].fillna(0.0)
            league_stats: Dict[str, float] = {}
            for column in numeric_columns:
                values = merged[column]
                mask = values.notna()
                if mask.any():
                    use_weights = weights_all[mask]
                    if use_weights.sum() > 0:
                        league_stats[column] = float(
                            np.average(values[mask], weights=use_weights)
                        )
                    else:
                        league_stats[column] = float(values[mask].mean())
                else:
                    league_stats[column] = np.nan
            positive_weight = float(weights_all[weights_all > 0].sum())
            if positive_weight <= 0:
                positive_weight = float(len(merged))
            league_stats["_weight"] = positive_weight
            league_baseline = pd.Series(league_stats)
        else:
            game_team_pos_baseline = pd.DataFrame()
            team_context_baseline = pd.DataFrame()
            team_pos_baseline = pd.DataFrame()
            pos_baseline = pd.DataFrame()
            league_baseline = pd.Series(dtype=float)

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

        def _assign_numeric_defaults(
            placeholder: Dict[str, Any], values: Optional[pd.Series]
        ) -> None:
            if values is None or not numeric_columns:
                return
            for column in numeric_columns:
                if column not in values:
                    continue
                value = values[column]
                if pd.isna(value):
                    continue
                if column not in placeholder or pd.isna(placeholder[column]):
                    placeholder[column] = value

        def _build_placeholder_row(lineup_row: pd.Series) -> Optional[Dict[str, Any]]:
            game_id_value = str(lineup_row.get("game_id", "")).strip()
            team_value = normalize_team_abbr(lineup_row.get("team"))
            position_value = normalize_position(lineup_row.get("position"))
            if not game_id_value or not team_value or not position_value:
                return None

            name_key = lineup_row.get("__pname_key", "") or ""
            if not name_key:
                name_seed = " ".join(
                    part
                    for part in [
                        str(lineup_row.get("first_name", "")).strip(),
                        str(lineup_row.get("last_name", "")).strip(),
                    ]
                    if part
                ) or str(lineup_row.get("player_name", "")).strip()
                name_key = robust_player_name_key(name_seed)
            if not name_key:
                return None

            template_pool = merged[
                (merged["game_id"].astype(str) == game_id_value)
                & (merged["team"] == team_value)
            ]
            if template_pool.empty:
                template_pool = merged[merged["game_id"].astype(str) == game_id_value]
            position_pool = pd.DataFrame()
            if template_pool.empty:
                base_values = {col: np.nan for col in merged.columns}
            else:
                position_pool = template_pool[template_pool["position"] == position_value]
                if position_pool.empty:
                    position_pool = template_pool
                base_row = position_pool.iloc[0]
                base_values = {col: base_row.get(col, np.nan) for col in merged.columns}
            fallback_candidates: List[pd.Series] = []
            if numeric_columns:
                if not position_pool.empty:
                    numeric_defaults = position_pool[numeric_columns].mean()
                    numeric_defaults = numeric_defaults.reindex(numeric_columns)
                    try:
                        pool_weight = float(
                            position_pool["_placeholder_weight"].fillna(0.0).sum()
                        )
                    except KeyError:
                        pool_weight = float(len(position_pool))
                    numeric_defaults.loc["_weight"] = pool_weight
                    fallback_candidates.append(numeric_defaults)

                def _append_baseline(
                    source: pd.DataFrame, key: Tuple[Any, ...]
                ) -> None:
                    if source is None or getattr(source, "empty", True):
                        return
                    try:
                        series = source.loc[key]
                    except KeyError:
                        return
                    if isinstance(series, pd.DataFrame):
                        if series.empty:
                            return
                        series = series.iloc[0]
                    fallback_candidates.append(series)

                _append_baseline(
                    team_context_baseline, (game_id_value, team_value)
                )
                _append_baseline(
                    game_team_pos_baseline, (game_id_value, team_value, position_value)
                )
                _append_baseline(team_pos_baseline, (team_value, position_value))
                _append_baseline(pos_baseline, (position_value,))
                if isinstance(league_baseline, pd.Series) and not league_baseline.empty:
                    fallback_candidates.append(league_baseline)

            player_name_value = str(lineup_row.get("player_name", "")).strip()
            if not player_name_value:
                first = str(lineup_row.get("first_name", "")).strip()
                last = str(lineup_row.get("last_name", "")).strip()
                player_name_value = " ".join(part for part in [first, last] if part)

            placeholder: Dict[str, Any] = dict(base_values)
            placeholder.pop("_placeholder_weight", None)
            placeholder["game_id"] = game_id_value
            placeholder["team"] = team_value
            if "opponent" in placeholder and pd.isna(placeholder["opponent"]):
                opp_pool = merged[
                    (merged["game_id"].astype(str) == game_id_value)
                    & (merged["team"] != team_value)
                ]
                if not opp_pool.empty:
                    placeholder["opponent"] = opp_pool.iloc[0].get("team")
            placeholder["position"] = position_value
            placeholder["player_name"] = player_name_value
            if "player_name_norm" in placeholder:
                placeholder["player_name_norm"] = normalize_player_name(
                    player_name_value
                )
            placeholder["__pname_key"] = name_key

            raw_player_id = lineup_row.get("player_id")
            if isinstance(raw_player_id, str) and raw_player_id.strip():
                player_id_value = raw_player_id.strip()
            else:
                player_id_value = f"lineup_{team_value}_{name_key}"
            placeholder["player_id"] = player_id_value

            depth_rank_value = parse_depth_rank(lineup_row.get("rank"))
            placeholder["depth_rank"] = depth_rank_value
            starter_flag = 1 if self._is_lineup_starter(position_value, depth_rank_value) else 0
            placeholder["is_starter"] = starter_flag
            placeholder["_lineup_hit"] = True
            placeholder["is_placeholder"] = True

            for defaults in fallback_candidates:
                _assign_numeric_defaults(placeholder, defaults)

            status_bucket = lineup_row.get("status_bucket")
            practice_status = lineup_row.get("practice_status")
            if status_bucket:
                status_bucket = normalize_injury_status(status_bucket)
                practice_status = normalize_practice_status(practice_status)
            else:
                status_bucket, practice_status = interpret_playing_probability(
                    lineup_row.get("playing_probability")
                )
                status_bucket = normalize_injury_status(status_bucket)
                practice_status = normalize_practice_status(practice_status)
            placeholder["status_bucket"] = status_bucket
            placeholder["practice_status"] = practice_status
            if "injury_priority" in placeholder:
                placeholder["injury_priority"] = INJURY_STATUS_PRIORITY.get(
                    status_bucket, INJURY_STATUS_PRIORITY.get("other", 1)
                )
            if "practice_priority" in placeholder:
                placeholder["practice_priority"] = PRACTICE_STATUS_PRIORITY.get(
                    practice_status, PRACTICE_STATUS_PRIORITY.get("available", 1)
                )

            updated_at = lineup_row.get("updated_at")
            if "updated_at" in placeholder:
                placeholder["updated_at"] = updated_at
            game_start_value = lineup_row.get("game_start")
            if "game_start" in placeholder:
                placeholder["game_start"] = game_start_value
            if "first_name" in placeholder:
                placeholder["first_name"] = lineup_row.get("first_name", "")
            if "last_name" in placeholder:
                placeholder["last_name"] = lineup_row.get("last_name", "")
            if "source" in placeholder and not placeholder.get("source"):
                placeholder["source"] = "msf-lineup"
            if "is_projected_starter" in placeholder:
                placeholder["is_projected_starter"] = True

            return placeholder

        if respect_lineups and not lineup_roster_full.empty:
            normalized_lineup = lineup_roster_full.copy()
            normalized_lineup["game_id"] = normalized_lineup["game_id"].astype(str)
            normalized_lineup["team"] = normalized_lineup["team"].apply(normalize_team_abbr)
            normalized_lineup["position"] = normalized_lineup["position"].apply(normalize_position)
            if "__pname_key" not in normalized_lineup.columns:
                normalized_lineup["__pname_key"] = normalized_lineup["player_name"].map(
                    robust_player_name_key
                )
            normalized_lineup["__pname_key"] = normalized_lineup["__pname_key"].fillna("")
            normalized_lineup = normalized_lineup[normalized_lineup["__pname_key"] != ""]

            existing_lineup_keys: Set[Tuple[str, str, str, str]] = set(
                zip(
                    merged["game_id"].astype(str),
                    merged["team"],
                    merged["__pname_key"],
                    merged["position"].apply(normalize_position),
                )
            )

            placeholder_rows: List[Dict[str, Any]] = []
            for _, lineup_row in normalized_lineup.iterrows():
                key = (
                    lineup_row.get("game_id", ""),
                    lineup_row.get("team"),
                    lineup_row.get("__pname_key", ""),
                    normalize_position(lineup_row.get("position")),
                )
                if key in existing_lineup_keys:
                    continue
                placeholder = _build_placeholder_row(lineup_row)
                if placeholder is None:
                    continue
                placeholder_rows.append(placeholder)
                existing_lineup_keys.add(key)

            if placeholder_rows:
                placeholder_df = pd.DataFrame(placeholder_rows)
                merged = safe_concat([merged, placeholder_df], ignore_index=True, sort=False)

        merged["_lineup_hit"] = merged["depth_rank"].notna()

        if respect_lineups and not lineup_audit_frame.empty:
            self._audit_lineup_matches(lineup_audit_frame, player_df, merged)

        candidate_pool = merged.copy()
        initial_candidate_count = len(candidate_pool)

        allowed_lineup_keys = locals().get("allowed_lineup_keys_all", set())
        starter_lineup_keys = locals().get("allowed_lineup_keys_starters", set())

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
        else:
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

        merged.loc[:, "depth_rank"] = merged["depth_rank"].fillna(9).astype(int)
        merged.loc[:, "is_starter"] = merged["is_starter"].fillna(0).astype(int)

        merged_before_filter = merged.copy()

        if respect_lineups:
            matched_count = int(merged_before_filter["_lineup_hit"].sum())
            logging.info(
                "Roster gate respected lineups: kept %d of %d players (matched=%d)",
                len(merged_before_filter[merged_before_filter["_lineup_hit"]]),
                initial_candidate_count,
                matched_count,
            )
            merged = merged_before_filter[merged_before_filter["_lineup_hit"]].copy()
        else:
            merged = merged_before_filter

        merged["_usage_confidence"] = compute_recency_usage_weights(merged)
        merged = merged.drop(columns=["_placeholder_weight"], errors="ignore")
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
            for _, row in lineup.iterrows():
                game_id_value = str(row.get("game_id"))
                team_value = row.get("team")
                pname_key_value = row.get("__pname_key", "")
                position_value = normalize_position(row.get("position", ""))

                key = (game_id_value, team_value, pname_key_value, position_value)
                if key in matched_keys:
                    continue
                summary_key = (game_id_value, team_value, pname_key_value)
                if summary_key in reported:
                    continue
                team_pool = players[
                    (players["game_id"] == game_id_value)
                    & (players["team"] == team_value)
                ]
                reasons: List[str] = []
                if team_pool.empty:
                    reasons.append("team missing in features")
                else:
                    name_pool = team_pool[team_pool["__pname_key"] == pname_key_value]
                    if name_pool.empty:
                        reasons.append("not in latest_players")
                    else:
                        pos_pool = name_pool[
                            name_pool["position"].apply(normalize_position)
                            == position_value
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

                player_label = str(row.get("player_name", "")).strip()
                if not player_label:
                    first = str(row.get("first_name", "")).strip()
                    last = str(row.get("last_name", "")).strip()
                    player_label = " ".join(
                        part for part in [first, last] if part
                    ).strip()
                player_label = player_label or pname_key_value or "(unknown)"

                logging.warning(
                    "[%s %s %s-%s] %s: %s",
                    game_id_value,
                    team_value,
                    position_value,
                    row.get("rank", ""),
                    player_label,
                    ", ".join(reasons),
                )
                reported.add(summary_key)
        except Exception:
            logging.debug("Lineup audit diagnostics failed", exc_info=True)

    def _compute_target_priors(self, df: pd.DataFrame, target: str) -> Dict[str, Any]:
        priors: Dict[str, Any] = {
            "league": {"mean": np.nan, "weight": 0.0, "q05": np.nan, "q95": np.nan},
            "position": {},
            "team_position": {},
        }
        if df.empty or target not in df.columns:
            return priors

        working = df.copy()
        if "team" in working.columns:
            working["team"] = working["team"].apply(normalize_team_abbr)
        if "position" in working.columns:
            working["position"] = working["position"].apply(normalize_position)

        mask_actual = working[target].notna()
        if "is_synthetic" in working.columns:
            mask_actual &= ~working["is_synthetic"].astype(bool)
        actual = working[mask_actual]
        if actual.empty:
            return priors

        weights_all = (
            actual.get("sample_weight", pd.Series(1.0, index=actual.index))
            .astype(float)
            .clip(lower=1e-4)
        )
        values_all = actual[target].astype(float)
        def _safe_weighted_quantiles(values: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
            try:
                sorter = np.argsort(values)
                values_sorted = values[sorter]
                weights_sorted = weights[sorter]
                cumulative = np.cumsum(weights_sorted)
                if cumulative[-1] == 0:
                    return (np.nan, np.nan)
                cumulative = cumulative / cumulative[-1]
                q05 = float(np.interp(0.05, cumulative, values_sorted))
                q95 = float(np.interp(0.95, cumulative, values_sorted))
                return (q05, q95)
            except Exception:
                return (np.nan, np.nan)

        q05_league, q95_league = _safe_weighted_quantiles(values_all.to_numpy(), weights_all.to_numpy())
        priors["league"] = {
            "mean": float(np.average(values_all, weights=weights_all)),
            "weight": float(weights_all.sum()),
            "q05": q05_league,
            "q95": q95_league,
        }

        for (team, position), group in actual.groupby(["team", "position"]):
            group_weights = (
                group.get("sample_weight", pd.Series(1.0, index=group.index))
                .astype(float)
                .clip(lower=1e-4)
            )
            group_values = group[target].astype(float)
            q05, q95 = _safe_weighted_quantiles(group_values.to_numpy(), group_weights.to_numpy())
            priors["team_position"][(team, position)] = {
                "mean": float(np.average(group_values, weights=group_weights)),
                "weight": float(group_weights.sum()),
                "q05": q05,
                "q95": q95,
            }

        for position, group in actual.groupby(["position"]):
            group_weights = (
                group.get("sample_weight", pd.Series(1.0, index=group.index))
                .astype(float)
                .clip(lower=1e-4)
            )
            group_values = group[target].astype(float)
            q05, q95 = _safe_weighted_quantiles(group_values.to_numpy(), group_weights.to_numpy())
            priors["position"][position] = {
                "mean": float(np.average(group_values, weights=group_weights)),
                "weight": float(group_weights.sum()),
                "q05": q05,
                "q95": q95,
            }

        return priors

    def _resolve_prior(
        self,
        target: str,
        team: Optional[str],
        position: Optional[str],
    ) -> Tuple[float, float]:
        priors = self.target_priors.get(target)
        if not priors:
            return (np.nan, 0.0)

        team_norm = normalize_team_abbr(team) if team else None
        pos_norm = normalize_position(position) if position else None

        if team_norm and pos_norm:
            entry = priors["team_position"].get((team_norm, pos_norm))
            if entry:
                return (entry.get("mean", np.nan), entry.get("weight", 0.0))

        if pos_norm:
            entry = priors["position"].get(pos_norm)
            if entry:
                return (entry.get("mean", np.nan), entry.get("weight", 0.0))

        league_entry = priors.get("league", {})
        return (
            league_entry.get("mean", np.nan),
            league_entry.get("weight", 0.0),
        )

    def _resolve_prior_bounds(
        self,
        target: str,
        team: Optional[str],
        position: Optional[str],
    ) -> Tuple[float, float]:
        priors = self.target_priors.get(target)
        if not priors:
            return (np.nan, np.nan)

        team_norm = normalize_team_abbr(team) if team else None
        pos_norm = normalize_position(position) if position else None

        if team_norm and pos_norm:
            entry = priors["team_position"].get((team_norm, pos_norm))
            if entry:
                return (entry.get("q05", np.nan), entry.get("q95", np.nan))

        if pos_norm:
            entry = priors["position"].get(pos_norm)
            if entry:
                return (entry.get("q05", np.nan), entry.get("q95", np.nan))

        league_entry = priors.get("league", {})
        return (league_entry.get("q05", np.nan), league_entry.get("q95", np.nan))

    def _build_neighbor_engine(
        self,
        transformer: ColumnTransformer,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        train_df: pd.DataFrame,
        feature_columns: List[str],
        train_weights: pd.Series,
        target: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            mask = (~train_df.get("is_synthetic", False).astype(bool)) & y_train.notna()
        except Exception:
            mask = y_train.notna()

        if not mask.any():
            return None

        feature_subset = X_train.loc[mask, feature_columns]
        if feature_subset.empty:
            return None

        try:
            transformed = transformer.transform(feature_subset)
        except Exception:
            logging.debug("Failed to transform features for neighbor prior on %s", target, exc_info=True)
            return None

        if hasattr(transformed, "toarray"):
            matrix = transformed.toarray()
        else:
            matrix = np.asarray(transformed)

        if matrix.shape[0] < 3:
            return None

        neighbor_count = int(min(25, matrix.shape[0]))
        try:
            nn = NearestNeighbors(n_neighbors=neighbor_count)
            nn.fit(matrix)
        except Exception:
            logging.debug("Unable to fit neighbor model for %s", target, exc_info=True)
            return None

        weight_array = (
            train_weights.loc[mask]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(1.0)
            .clip(lower=1e-4)
            .to_numpy()
        )
        target_array = y_train.loc[mask].astype(float).to_numpy()

        smoothing = float(np.median(weight_array) * neighbor_count)
        smoothing = max(smoothing, 5.0)

        return {
            "transformer": transformer,
            "feature_columns": list(feature_columns),
            "nn": nn,
            "targets": target_array,
            "weights": weight_array,
            "smoothing": smoothing,
            "n_neighbors": neighbor_count,
        }

    def calibrate_player_predictions(
        self,
        target: str,
        feature_slice: pd.DataFrame,
        predictions: np.ndarray,
    ) -> np.ndarray:
        if feature_slice is None or feature_slice.empty:
            return predictions

        preds = np.asarray(predictions, dtype=float)
        if preds.size == 0:
            return preds

        special_entry = self.special_models.get(target, {}) if hasattr(self, "special_models") else {}
        calibration_info = special_entry.get("calibration") if isinstance(special_entry, dict) else None
        if calibration_info and calibration_info.get("method") == "logistic":
            try:
                model = calibration_info.get("model")
                if model is not None:
                    preds = model.predict_proba(preds.reshape(-1, 1))[:, 1]
            except Exception:
                logging.debug("Unable to apply stored calibration for %s", target, exc_info=True)

        features = feature_slice.copy()
        if "team" in features.columns:
            features["team"] = features["team"].apply(normalize_team_abbr)
        if "position" in features.columns:
            features["position"] = features["position"].apply(normalize_position)

        usage_conf = features.get("_usage_confidence", pd.Series(0.5, index=features.index))
        usage_conf = pd.to_numeric(usage_conf, errors="coerce").fillna(0.5).clip(0.0, 1.0)
        is_placeholder = features.get("is_placeholder", pd.Series(False, index=features.index))
        is_placeholder = coerce_boolean_mask(is_placeholder)

        neighbor_engine = self.prior_engines.get(target)
        neighbor_means = np.full(len(features), np.nan, dtype=float)
        neighbor_supports = np.zeros(len(features), dtype=float)
        neighbor_strengths = np.zeros(len(features), dtype=float)

        if neighbor_engine:
            required_cols = neighbor_engine.get("feature_columns", [])
            if required_cols:
                aligned = features.reindex(columns=required_cols, fill_value=np.nan)
                try:
                    transformed = neighbor_engine["transformer"].transform(aligned)
                    if hasattr(transformed, "toarray"):
                        matrix = transformed.toarray()
                    else:
                        matrix = np.asarray(transformed)
                    distances, indices = neighbor_engine["nn"].kneighbors(
                        matrix, return_distance=True
                    )
                    neighbor_targets = neighbor_engine["targets"][indices]
                    base_weights = neighbor_engine["weights"][indices]
                    inv_distance = 1.0 / (distances + 1e-6)
                    weighted = base_weights * inv_distance
                    weight_sums = weighted.sum(axis=1)
                    valid_mask = weight_sums > 0
                    if np.any(valid_mask):
                        neighbor_means[valid_mask] = (
                            (weighted[valid_mask] * neighbor_targets[valid_mask]).sum(axis=1)
                            / weight_sums[valid_mask]
                        )
                        neighbor_supports[valid_mask] = weight_sums[valid_mask]
                        smoothing = float(neighbor_engine.get("smoothing", 10.0))
                        neighbor_strengths[valid_mask] = neighbor_supports[valid_mask] / (
                            neighbor_supports[valid_mask] + smoothing
                        )
                except Exception:
                    logging.debug(
                        "Unable to apply neighbor prior for %s during calibration", target, exc_info=True
                    )

        for idx, (row_idx, row) in enumerate(features.iterrows()):
            confidence = (
                float(usage_conf.loc[row_idx]) if row_idx in usage_conf.index else 0.5
            )
            placeholder_flag = (
                bool(is_placeholder.loc[row_idx]) if row_idx in is_placeholder.index else False
            )

            # Skip calibration when we already have a confident, data-backed projection.
            if not placeholder_flag and confidence >= 0.6:
                continue

            prior_mean, prior_weight = self._resolve_prior(
                target,
                row.get("team"),
                row.get("position"),
            )

            neighbor_mean = neighbor_means[idx]
            neighbor_conf = neighbor_strengths[idx]
            neighbor_weight = neighbor_supports[idx]

            combined_mean = np.nan
            combined_weight = 0.0

            if not np.isnan(neighbor_mean) and neighbor_conf > 0:
                neighbor_support = max(neighbor_weight, 0.0)
                if not np.isnan(prior_mean) and prior_weight > 0:
                    combined_mean = (
                        neighbor_mean * neighbor_support + prior_mean * prior_weight
                    ) / (neighbor_support + prior_weight)
                    combined_weight = neighbor_support + prior_weight
                else:
                    combined_mean = neighbor_mean
                    combined_weight = neighbor_support
            elif not np.isnan(prior_mean) and prior_weight > 0:
                combined_mean = prior_mean
                combined_weight = prior_weight
            else:
                combined_mean = neighbor_mean if not np.isnan(neighbor_mean) else prior_mean
                if np.isnan(combined_mean):
                    continue

            prior_strength = 0.0
            if combined_weight > 0:
                prior_strength = combined_weight / (combined_weight + 25.0)

            mix_strength = max(prior_strength, neighbor_conf)

            if placeholder_flag:
                base_alpha = max(0.35, 1.0 - confidence) * mix_strength
                alpha = np.clip(base_alpha, 0.0, 0.45)
            else:
                base_alpha = (1.0 - confidence) * mix_strength
                alpha = np.clip(base_alpha, 0.0, 0.25)

            if alpha <= 0:
                continue

            current_pred = preds[idx]
            if np.isnan(current_pred) or np.isinf(current_pred):
                preds[idx] = combined_mean
            else:
                preds[idx] = (1 - alpha) * current_pred + alpha * combined_mean

            lower_q, upper_q = self._resolve_prior_bounds(
                target,
                row.get("team"),
                row.get("position"),
            )
            if not np.isnan(lower_q) and preds[idx] < lower_q:
                preds[idx] = lower_q
            if not np.isnan(upper_q) and preds[idx] > upper_q:
                preds[idx] = upper_q

        if target in NON_NEGATIVE_TARGETS:
            np.maximum(preds, 0.0, out=preds)

        return preds

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

            prepared = self._prepare_regression_training_data(df, target)
            if prepared is None:
                continue

            sorted_df, feature_columns, weight_series = prepared
            try:
                model, _summary = self._train_regression_model(
                    sorted_df,
                    feature_columns,
                    target,
                    weight_series=weight_series,
                )
            except RuntimeError:
                raise

            if model is not None:
                models[target] = model
        return models

    def _prepare_regression_training_data(
        self, df: pd.DataFrame, target: str
    ) -> Optional[Tuple[pd.DataFrame, List[str], Optional[pd.Series]]]:
        if len(df) < 20 or target not in df.columns or df[target].nunique() <= 1:
            logging.warning(
                "Not enough data to train %s model (rows=%d, unique targets=%d).",
                target,
                len(df),
                df[target].nunique() if target in df.columns else 0,
            )
            return None

        df = df.copy()
        if "sample_weight" not in df.columns:
            df["sample_weight"] = 1.0
        else:
            df["sample_weight"] = (
                pd.to_numeric(df["sample_weight"], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(1.0)
                .clip(lower=1e-4)
            )
        if "is_synthetic" not in df.columns:
            df["is_synthetic"] = False
        else:
            df["is_synthetic"] = df["is_synthetic"].fillna(False).astype(bool)

        self.target_priors[target] = self._compute_target_priors(df, target)

        ignore_columns = {target, "sample_weight", "is_synthetic"}
        feature_columns: List[str] = []
        for col in df.columns:
            if col in ignore_columns:
                continue
            series = df[col]
            if not pd.api.types.is_numeric_dtype(series):
                continue
            if not series.notna().any():
                continue
            feature_columns.append(col)

        if target.endswith("_win_prob"):
            filtered_features: List[str] = []
            removal_tokens = ("moneyline", "implied_prob", "line_", "spread")
            for col in feature_columns:
                lower = col.lower()
                if any(token in lower for token in removal_tokens):
                    continue
                filtered_features.append(col)
            if filtered_features:
                feature_columns = filtered_features

        if not feature_columns:
            logging.warning("No usable numeric features available to train %s model; skipping.", target)
            return None

        sorted_df = self._sort_by_time(df).reset_index(drop=True)
        weight_series = sorted_df.get("sample_weight")
        self.prior_engines[target] = None
        return sorted_df, feature_columns, weight_series


    def _train_regression_model(
        self,
        sorted_df: pd.DataFrame,
        feature_columns: List[str],
        target: str,
        weight_series: Optional[pd.Series] = None,
        *,
        vig_threshold: float = 0.02,
        min_edge: float = 0.0,
    ) -> Tuple[Pipeline, Dict[str, Any]]:
        """
        Walk-forward, season/week-ordered evaluation with betting diagnostics.
        Aborts (raises RuntimeError) if ROI lower CI bound <= vig_threshold.
        """
        import math
        from sklearn.base import clone
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        df = sorted_df.copy()

        def _american_to_prob(series: pd.Series) -> pd.Series:
            odds = pd.to_numeric(series, errors="coerce")
            prob = pd.Series(np.nan, index=odds.index, dtype=float)
            if odds is None:
                return prob
            positive = odds >= 0
            prob.loc[positive] = 100.0 / (odds.loc[positive] + 100.0)
            prob.loc[~positive] = (-odds.loc[~positive]) / ((-odds.loc[~positive]) + 100.0)
            return prob

        def _ensure_market_data(frame: pd.DataFrame) -> pd.DataFrame:
            working = frame.copy()
            for side in ("home", "away"):
                other_side = "away" if side == "home" else "home"
                moneyline_col = f"{side}_moneyline"
                implied_col = f"{side}_implied_prob"
                fair_col = f"{side}_fair_prob"
                closing_money_col = f"{side}_closing_moneyline"
                closing_prob_col = f"{side}_closing_implied_prob"

                if moneyline_col not in working.columns:
                    working[moneyline_col] = np.nan
                working[moneyline_col] = pd.to_numeric(
                    working.get(moneyline_col), errors="coerce"
                )

                if closing_money_col in working.columns:
                    closing_series = pd.to_numeric(
                        working.get(closing_money_col), errors="coerce"
                    )
                    working[moneyline_col] = closing_series.combine_first(
                        working[moneyline_col]
                    )

                if implied_col not in working.columns:
                    working[implied_col] = np.nan
                working[implied_col] = pd.to_numeric(
                    working.get(implied_col), errors="coerce"
                )

                if closing_prob_col in working.columns:
                    closing_prob_series = pd.to_numeric(
                        working.get(closing_prob_col), errors="coerce"
                    )
                    working[implied_col] = closing_prob_series.combine_first(
                        working[implied_col]
                    )

                derived_probs = _american_to_prob(working[moneyline_col])
                working[implied_col] = working[implied_col].fillna(derived_probs)

                partner_col = f"{other_side}_implied_prob"
                if partner_col in working.columns:
                    partner_probs = pd.to_numeric(working[partner_col], errors="coerce")
                    missing_mask = working[implied_col].isna() & partner_probs.notna()
                    if missing_mask.any():
                        working.loc[missing_mask, implied_col] = 1.0 - partner_probs.loc[missing_mask]

                working[fair_col] = np.nan

            if {"home_implied_prob", "away_implied_prob"}.issubset(working.columns):
                home_probs = pd.to_numeric(working["home_implied_prob"], errors="coerce")
                away_probs = pd.to_numeric(working["away_implied_prob"], errors="coerce")

                missing_home = home_probs.isna() & away_probs.notna()
                if missing_home.any():
                    home_probs.loc[missing_home] = 1.0 - away_probs.loc[missing_home]

                missing_away = away_probs.isna() & home_probs.notna()
                if missing_away.any():
                    away_probs.loc[missing_away] = 1.0 - home_probs.loc[missing_away]

                home_probs = home_probs.clip(lower=0.0, upper=1.0)
                away_probs = away_probs.clip(lower=0.0, upper=1.0)

                working["home_implied_prob"] = home_probs
                working["away_implied_prob"] = away_probs

                total = home_probs + away_probs
                valid_total = total > 0
                working["home_fair_prob"] = np.nan
                working["away_fair_prob"] = np.nan
                if valid_total.any():
                    with np.errstate(divide="ignore", invalid="ignore"):
                        working.loc[valid_total, "home_fair_prob"] = (
                            home_probs.loc[valid_total] / total.loc[valid_total]
                        )
                        working.loc[valid_total, "away_fair_prob"] = (
                            away_probs.loc[valid_total] / total.loc[valid_total]
                        )
            else:
                working["home_fair_prob"] = np.nan
                working["away_fair_prob"] = np.nan

            return working

        if target.endswith("_win_prob"):
            df = _ensure_market_data(df)

        def _as_weight_array(series: Optional[pd.Series]) -> Optional[np.ndarray]:
            if series is None:
                return None
            if isinstance(series, pd.Series):
                arr = (
                    pd.to_numeric(series, errors="coerce")
                    .fillna(1.0)
                    .clip(lower=1e-6)
                    .astype(float)
                    .to_numpy()
                )
            else:
                arr = np.asarray(series, dtype=float)
                arr = np.where(np.isfinite(arr), arr, 1.0)
                arr = np.clip(arr, 1e-6, None)
            return arr

        def _weighted_metrics(
            y_true: Union[pd.Series, np.ndarray],
            y_pred: Union[pd.Series, np.ndarray],
            weights: Optional[Union[pd.Series, np.ndarray]],
        ) -> Tuple[float, float, float]:
            from math import sqrt
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            y_t = np.asarray(y_true, dtype=float)
            y_p = np.asarray(y_pred, dtype=float)
            w = None if weights is None else np.asarray(weights, dtype=float)
            if w is not None and w.shape != y_t.shape:
                w = None
            r2 = float(r2_score(y_t, y_p, sample_weight=w)) if y_t.size else float("nan")
            mae = float(mean_absolute_error(y_t, y_p, sample_weight=w)) if y_t.size else float("nan")
            if y_t.size:
                mse = mean_squared_error(y_t, y_p, sample_weight=w)
                rmse = float(sqrt(mse))
            else:
                rmse = float("nan")
            return r2, mae, rmse

        # Require time keys
        if not {"season", "week"}.issubset(df.columns):
            raise ValueError(f"{target}: requires 'season' and 'week' columns for walk-forward CV")

        # Build a clean matrix
        X_all = df[feature_columns]
        y_all = df[target]
        w_all = _as_weight_array(weight_series)  # existing helper

        # Simple preprocessor; keep your original encoder if you had one upstream
        preprocessor = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=False)),
            ]
        )

        base_est = GradientBoostingRegressor(random_state=42)
        model = Pipeline([("preprocessor", preprocessor), ("regressor", base_est)])

        # ---- Walk-forward splits by unique (season, week) keys (expanding window) ----
        df["_sw"] = list(zip(df["season"].astype(str), df["week"].astype(int, errors="ignore").fillna(0).astype(int)))
        sw_keys = pd.Series(df["_sw"].unique().tolist())
        # sort season/week by real world order: season asc, week asc
        sw_keys = sw_keys.sort_values(key=lambda s: s.map(lambda x: (x[0], x[1]))).tolist()

        # at least 3 folds: train on >=1, validate on the next, keep rolling
        folds = []
        for i in range(1, len(sw_keys) - 1):
            train_keys = set(sw_keys[:i])        # everything strictly before validation
            valid_key = sw_keys[i]               # one time-slice as validation
            train_idx = df["_sw"].isin(train_keys).values
            valid_idx = (df["_sw"] == valid_key).values
            if train_idx.sum() >= 120 and valid_idx.sum() >= 30:  # reasonable sample minimums
                folds.append((train_idx, valid_idx))

        per_fold = []
        fold_details: List[Dict[str, Any]] = []
        idx_all: List[np.ndarray] = []
        used_walk_forward = True

        if not folds:
            # Fallback to a simple chronological holdout when there are not enough
            # season/week buckets to run the expanding walk-forward evaluation.
            logging.warning(
                "%s: insufficient time slices for walk-forward; fitting single model with holdout diagnostics.",
                target,
            )

            valid_size = max(1, min(max(len(df) // 5, 10), len(df) // 2))
            train_size = len(df) - valid_size
            if train_size <= 0:
                train_size = max(len(df) - 1, 1)
                valid_size = len(df) - train_size

            train_idx = np.zeros(len(df), dtype=bool)
            valid_idx = np.zeros(len(df), dtype=bool)
            train_idx[:train_size] = True
            valid_idx[train_size:train_size + valid_size] = True

            folds = [(train_idx, valid_idx)]
            used_walk_forward = False

        # ---- Run walk-forward (or fallback holdout) evaluation ----
        for k, (tr_idx, va_idx) in enumerate(folds, 1):
            X_tr, y_tr = X_all.loc[tr_idx], y_all.loc[tr_idx]
            X_va, y_va = X_all.loc[va_idx], y_all.loc[va_idx]
            w_tr = w_all[tr_idx] if w_all is not None else None
            w_va = w_all[va_idx] if w_all is not None else None

            est = clone(model)
            fit_kwargs = {"regressor__sample_weight": w_tr} if w_tr is not None else {}
            est.fit(X_tr, y_tr, **fit_kwargs)

            y_hat = est.predict(X_va)
            idx = np.where(va_idx)[0]
            idx_all.append(idx)

            fold_details.append(
                {
                    "fold": k,
                    "indices": idx,
                    "y_true": np.asarray(y_va, dtype=float),
                    "y_pred": np.asarray(y_hat, dtype=float),
                    "weights": np.asarray(w_va, dtype=float) if w_va is not None else None,
                }
            )

        if fold_details:
            preds_all = np.concatenate([detail["y_pred"] for detail in fold_details])
            idx_all = np.concatenate(idx_all)
        else:
            preds_all = np.array([])
            idx_all = np.array([])

        out_df = df.iloc[idx_all].copy() if idx_all.size else df.iloc[0:0].copy()
        if target.endswith("_win_prob"):
            out_df = _ensure_market_data(out_df)
        out_df["_y_pred"] = preds_all
        out_df["_y_pred_raw"] = preds_all
        out_df["_abs_err"] = (out_df[target] - out_df["_y_pred"]).abs()

        calibration_record: Optional[Dict[str, Any]] = None
        if fold_details and preds_all.size:
            y_true_all = np.concatenate([detail["y_true"] for detail in fold_details])
            weight_stacks = [
                detail["weights"]
                if detail["weights"] is not None
                else np.ones_like(detail["y_true"], dtype=float)
                for detail in fold_details
            ]
            weights_all = np.concatenate(weight_stacks) if weight_stacks else None

            if target.endswith("_win_prob") and np.unique(y_true_all).size > 1:
                try:
                    from sklearn.linear_model import LogisticRegression

                    lr = LogisticRegression(max_iter=1000)
                    X_cal = preds_all.reshape(-1, 1)
                    lr.fit(X_cal, y_true_all, sample_weight=weights_all)
                    calibrated = lr.predict_proba(X_cal)[:, 1]

                    start = 0
                    for detail in fold_details:
                        fold_len = len(detail["y_pred"])
                        detail["final_pred"] = calibrated[start:start + fold_len]
                        start += fold_len

                    preds_all = calibrated
                    out_df["_y_pred"] = calibrated
                    out_df["_abs_err"] = (out_df[target] - out_df["_y_pred"]).abs()
                    calibration_record = {"method": "logistic", "model": lr}
                except Exception:
                    logging.debug("Failed to fit logistic calibration for %s", target, exc_info=True)

        if fold_details:
            for detail in fold_details:
                if "final_pred" not in detail:
                    detail["final_pred"] = detail["y_pred"]
                weights_fold = detail["weights"]
                r2, mae, rmse = _weighted_metrics(detail["y_true"], detail["final_pred"], weights_fold)
                per_fold.append(
                    {
                        "fold": detail["fold"],
                        "n": int(len(detail["y_true"])),
                        "r2": r2,
                        "mae": float(mae),
                        "rmse": float(rmse),
                    }
                )

        # ---- Betting diagnostics ----
        # 1) MAE vs "closing lines" (if present)
        closing_cols = [
            c
            for c in out_df.columns
            if c.startswith("line_") or c.endswith("_implied_prob")
        ]
        closing_cols.sort(key=lambda col: (0 if "closing" in col else 1, col))
        mae_vs_closing = None
        baseline_reference: Optional[pd.Series] = None
        if not target.endswith("_win_prob") and not out_df.empty:
            try:
                baseline_values: List[float] = []
                for idx, row in out_df.iterrows():
                    prior_mean, _ = self._resolve_prior(
                        target,
                        row.get("team"),
                        row.get("position"),
                    )
                    if math.isnan(prior_mean):
                        league_prior = self.target_priors.get(target, {}).get("league", {})
                        prior_mean = float(league_prior.get("mean", np.nan))
                    baseline_values.append(float(prior_mean))
                baseline_reference = pd.Series(baseline_values, index=out_df.index, dtype=float)
            except Exception:
                logging.debug("%s: unable to resolve baseline priors for diagnostics", target, exc_info=True)
                baseline_reference = None
            else:
                out_df["_baseline_reference"] = baseline_reference

        if target.endswith("_win_prob"):
            side = out_df.get("team_side")  # optional column you may carry ("home"/"away")
            if side is not None:
                home_ref = pd.Series(np.nan, index=out_df.index, dtype=float)
                away_ref = pd.Series(np.nan, index=out_df.index, dtype=float)
                if "home_closing_implied_prob" in out_df:
                    home_ref = pd.to_numeric(out_df["home_closing_implied_prob"], errors="coerce")
                if "away_closing_implied_prob" in out_df:
                    away_ref = pd.to_numeric(out_df["away_closing_implied_prob"], errors="coerce")
                if "home_implied_prob" in out_df:
                    home_ref = home_ref.combine_first(
                        pd.to_numeric(out_df["home_implied_prob"], errors="coerce")
                    )
                if "away_implied_prob" in out_df:
                    away_ref = away_ref.combine_first(
                        pd.to_numeric(out_df["away_implied_prob"], errors="coerce")
                    )
                if home_ref.notna().any() or away_ref.notna().any():
                    cl = np.where(side == "home", home_ref, away_ref)
                    mae_vs_closing = float(
                        np.nanmean(np.abs(out_df["_y_pred"] - pd.to_numeric(cl, errors="coerce")))
                    )
        elif closing_cols:
            # e.g., player props with 'line_receiving_yards', etc.
            # Choose the first matching line as a reference
            ref = pd.to_numeric(out_df[closing_cols[0]], errors="coerce")
            if ref.notna().any():
                mae_vs_closing = float(
                    np.nanmean(np.abs(out_df["_y_pred"] - ref.astype(float)))
                )

        if mae_vs_closing is None and baseline_reference is not None:
            if baseline_reference.notna().any():
                mae_vs_closing = float(
                    np.nanmean(np.abs(out_df["_y_pred"] - baseline_reference.astype(float)))
                )

        # 2) EV rule → pick bets → compute hit rate & ROI (+ CI)
        def _american_to_payout(odds: float) -> float:
            # Net profit per unit staked if it wins
            if odds >= 0:
                return odds / 100.0
            return 100.0 / (-odds)

        # For game win-prob targets
        roi, roi_lo, roi_hi, hit_rate, n_bets = None, None, None, None, 0
        edge_source = None

        def _evaluate_edges(preds_array: np.ndarray, label: str) -> Optional[Dict[str, Any]]:
            if preds_array.size != len(out_df) or not len(out_df):
                return None

            pred_series = pd.Series(preds_array, index=out_df.index, dtype=float)

            required_score_cols = {"home_score", "away_score"}
            missing_scores = required_score_cols - set(out_df.columns)
            if missing_scores:
                logging.debug(
                    "%s: skipping edge evaluation (score columns missing: %s)",
                    target,
                    ", ".join(sorted(missing_scores)),
                )
                return None

            def _merge_price(side: str) -> pd.Series:
                candidates: List[pd.Series] = []
                for col in (f"{side}_closing_moneyline", f"{side}_moneyline"):
                    if col in out_df.columns:
                        candidates.append(pd.to_numeric(out_df[col], errors="coerce"))
                if not candidates:
                    return pd.Series(np.nan, index=out_df.index, dtype=float)
                series = candidates[0]
                for extra in candidates[1:]:
                    series = series.combine_first(extra)
                return series

            home_prices = _merge_price("home")
            away_prices = _merge_price("away")

            if home_prices.notna().sum() == 0 and away_prices.notna().sum() == 0:
                logging.debug(
                    "%s: skipping edge evaluation (no recorded sportsbook moneylines)",
                    target,
                )
                return None

            home_probs = _american_to_prob(home_prices)
            away_probs = _american_to_prob(away_prices)

            # Fill missing implied probabilities with the complement if only one side is known
            missing_home = home_probs.isna() & away_probs.notna()
            if missing_home.any():
                home_probs.loc[missing_home] = 1.0 - away_probs.loc[missing_home].clip(0.0, 1.0)

            missing_away = away_probs.isna() & home_probs.notna()
            if missing_away.any():
                away_probs.loc[missing_away] = 1.0 - home_probs.loc[missing_away].clip(0.0, 1.0)

            # Require completed games for ROI computation
            results_mask = (
                out_df["home_score"].notna() & out_df["away_score"].notna()
            )

            home_valid = home_prices.notna() & home_probs.notna()
            away_valid = away_prices.notna() & away_probs.notna()
            valid_mask = (home_valid | away_valid) & results_mask

            if not valid_mask.any():
                missing_without_odds = results_mask & ~(home_valid | away_valid)
                subset_cols = [
                    col for col in ["game_id", "season", "week"] if col in out_df.columns
                ]
                if subset_cols:
                    missing_rows = out_df.loc[missing_without_odds, subset_cols].drop_duplicates()
                else:
                    missing_rows = pd.DataFrame(index=out_df.index[missing_without_odds])
                if not missing_rows.empty:
                    logging.warning(
                        "%s: skipping ROI evaluation because %d games lack recorded moneylines. "
                        "Run the odds ingestion backfill to capture historical prices before betting decisions.",
                        target,
                        len(missing_rows),
                    )
                else:
                    logging.warning("%s: no rows with complete market data for edge check", target)
                return None

            missing_without_odds = results_mask & ~(home_valid | away_valid)
            if missing_without_odds.any():
                logging.info(
                    "%s: dropping %d completed games without stored moneylines from ROI calculation",
                    target,
                    int(missing_without_odds.sum()),
                )

            pred_series = pred_series.loc[valid_mask]
            home_probs = home_probs.loc[valid_mask]
            away_probs = away_probs.loc[valid_mask]
            home_prices = home_prices.loc[valid_mask]
            away_prices = away_prices.loc[valid_mask]
            home_valid = home_valid.loc[valid_mask]
            away_valid = away_valid.loc[valid_mask]

            total_prob = home_probs + away_probs
            fair_home = pd.Series(np.nan, index=home_probs.index, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                fair_mask = total_prob > 0
                fair_home.loc[fair_mask] = home_probs.loc[fair_mask] / total_prob.loc[fair_mask]
            fair_home = fair_home.clip(0.0, 1.0)
            fair_away = (1.0 - fair_home).clip(0.0, 1.0)

            # Use fair probabilities when available, otherwise fall back to implied values
            home_reference = fair_home.fillna(home_probs.clip(0.0, 1.0))
            away_reference = fair_away.fillna(away_probs.clip(0.0, 1.0))

            choose_home = pred_series - home_reference
            choose_away = (1.0 - pred_series) - away_reference

            def _build_picks_for_threshold(threshold: float) -> Optional[pd.DataFrame]:
                picks: List[pd.DataFrame] = []

                pick_home = home_valid & (choose_home > threshold)
                if pick_home.any():
                    home_index = pred_series.index[pick_home]
                    home_df = pd.DataFrame(index=home_index)
                    home_df["p_model"] = pred_series.loc[pick_home].clip(0.0, 1.0)
                    home_df["side_win"] = (
                        out_df.loc[home_index, "home_score"] > out_df.loc[home_index, "away_score"]
                    ).astype(int)
                    home_df["payout"] = home_prices.loc[pick_home].apply(_american_to_payout)
                    home_df["_edge"] = choose_home.loc[pick_home].astype(float)
                    home_df["_threshold"] = threshold
                    home_df["_side"] = "home"
                    home_df["_is_push"] = False
                    if "start_time" in out_df.columns:
                        home_df["_start_time"] = pd.to_datetime(
                            out_df.loc[home_index, "start_time"], errors="coerce"
                        )
                    home_columns = [
                        "p_model",
                        "side_win",
                        "payout",
                        "_edge",
                        "_threshold",
                        "_side",
                        "_is_push",
                    ]
                    if "_start_time" in home_df.columns:
                        home_columns.append("_start_time")
                    picks.append(
                        home_df[home_columns]
                    )

                pick_away = away_valid & (choose_away > threshold)
                if pick_away.any():
                    away_index = pred_series.index[pick_away]
                    away_df = pd.DataFrame(index=away_index)
                    away_df["p_model"] = (1.0 - pred_series.loc[pick_away]).clip(0.0, 1.0)
                    away_df["side_win"] = (
                        out_df.loc[away_index, "away_score"] > out_df.loc[away_index, "home_score"]
                    ).astype(int)
                    away_df["payout"] = away_prices.loc[pick_away].apply(_american_to_payout)
                    away_df["_edge"] = choose_away.loc[pick_away].astype(float)
                    away_df["_threshold"] = threshold
                    away_df["_side"] = "away"
                    away_df["_is_push"] = False
                    if "start_time" in out_df.columns:
                        away_df["_start_time"] = pd.to_datetime(
                            out_df.loc[away_index, "start_time"], errors="coerce"
                        )
                    away_columns = [
                        "p_model",
                        "side_win",
                        "payout",
                        "_edge",
                        "_threshold",
                        "_side",
                        "_is_push",
                    ]
                    if "_start_time" in away_df.columns:
                        away_columns.append("_start_time")
                    picks.append(
                        away_df[away_columns]
                    )

                if not picks:
                    return None
                return pd.concat(picks, ignore_index=True)

            candidate_thresholds: List[float] = []
            seen_thresholds: Set[float] = set()
            for candidate in [min_edge, (min_edge * 0.5 if min_edge > 0 else None), 0.0, -0.005]:
                if candidate is None:
                    continue
                candidate = float(candidate)
                if math.isnan(candidate):
                    continue
                if candidate not in seen_thresholds:
                    candidate_thresholds.append(candidate)
                    seen_thresholds.add(candidate)

            selected_threshold: Optional[float] = None
            picks_df: Optional[pd.DataFrame] = None
            for threshold in candidate_thresholds:
                candidate_df = _build_picks_for_threshold(threshold)
                if candidate_df is not None and not candidate_df.empty:
                    picks_df = candidate_df
                    selected_threshold = threshold
                    break

            if picks_df is None:
                fallback_df = _build_picks_for_threshold(float("-inf"))
                if fallback_df is not None and not fallback_df.empty:
                    fallback_df = fallback_df.sort_values("_edge", ascending=False)
                    top_n = max(1, min(5, len(fallback_df)))
                    picks_df = fallback_df.head(top_n).reset_index(drop=True)
                    selected_threshold = float("-inf")
                    logging.debug("%s: forcing %d bets using fallback top-edge selection", target, len(picks_df))

            if picks_df is None:
                logging.debug("%s: no picks available even after fallback thresholds", target)
                return None

            label_suffix = label
            if selected_threshold is not None:
                if selected_threshold == float("-inf"):
                    label_suffix = f"{label}|forced_top_edges"
                elif selected_threshold != min_edge:
                    label_suffix = f"{label}|thr={selected_threshold:+.3f}"

            bet_count = len(picks_df)
            returns = np.where(
                picks_df["_is_push"],
                0.0,
                np.where(picks_df["side_win"] == 1, picks_df["payout"].values, -1.0),
            )
            roi_val = float(np.mean(returns))
            m = returns.mean()
            s = returns.std(ddof=1) if bet_count > 1 else 0.0
            se = s / math.sqrt(max(bet_count, 1))
            z = 1.96
            roi_lo_val, roi_hi_val = float(m - z * se), float(m + z * se)
            non_push_mask = ~picks_df["_is_push"].astype(bool)
            if non_push_mask.any():
                hit_val = float(np.mean(picks_df.loc[non_push_mask, "side_win"]))
            else:
                hit_val = float("nan")

            recent_summary: Optional[Dict[str, Any]] = None
            if "_start_time" in picks_df.columns and picks_df["_start_time"].notna().any():
                valid_times = picks_df.loc[picks_df["_start_time"].notna(), "_start_time"]
                latest_time = valid_times.max()
                if isinstance(latest_time, dt.datetime):
                    recent_cutoff = latest_time - dt.timedelta(weeks=8)
                    recent_df = picks_df.loc[picks_df["_start_time"] >= recent_cutoff]
                    if len(recent_df) >= 10:
                        recent_returns = np.where(
                            recent_df["_is_push"],
                            0.0,
                            np.where(
                                recent_df["side_win"] == 1,
                                recent_df["payout"].values,
                                -1.0,
                            ),
                        )
                        recent_roi = float(np.mean(recent_returns))
                        r_mean = recent_returns.mean()
                        r_std = (
                            recent_returns.std(ddof=1)
                            if len(recent_returns) > 1
                            else 0.0
                        )
                        r_se = r_std / math.sqrt(max(len(recent_returns), 1))
                        r_lo, r_hi = float(r_mean - z * r_se), float(r_mean + z * r_se)
                        non_push_recent = ~recent_df["_is_push"].astype(bool)
                        if non_push_recent.any():
                            recent_hit = float(
                                np.mean(recent_df.loc[non_push_recent, "side_win"])
                            )
                        else:
                            recent_hit = float("nan")
                        recent_summary = {
                            "roi": recent_roi,
                            "roi_lo": r_lo,
                            "roi_hi": r_hi,
                            "hit_rate": recent_hit,
                            "n_bets": int(len(recent_df)),
                            "window_start": recent_cutoff,
                            "window_end": latest_time,
                        }

            return {
                "roi": roi_val,
                "roi_lo": roi_lo_val,
                "roi_hi": roi_hi_val,
                "hit_rate": hit_val,
                "n_bets": bet_count,
                "label": label_suffix,
                "recent": recent_summary,
            }

        recent_metrics: Optional[Dict[str, Any]] = None

        if target.endswith("_win_prob") and {"home_moneyline", "away_moneyline"}.issubset(out_df.columns):
            prediction_candidates: List[Tuple[str, np.ndarray]] = []
            if len(out_df):
                base_label = "calibrated" if calibration_record is not None else "model"
                prediction_candidates.append((base_label, out_df["_y_pred"].to_numpy(dtype=float)))
                if "_y_pred_raw" in out_df.columns:
                    prediction_candidates.append(("raw", out_df["_y_pred_raw"].to_numpy(dtype=float)))

            best_option: Optional[Dict[str, Any]] = None
            seen_labels: set = set()
            for label, preds_array in prediction_candidates:
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                evaluation = _evaluate_edges(preds_array, label)
                if evaluation is None:
                    continue
                if best_option is None or evaluation["roi"] > best_option["roi"]:
                    best_option = evaluation

            if best_option is not None:
                roi = best_option["roi"]
                roi_lo = best_option["roi_lo"]
                roi_hi = best_option["roi_hi"]
                hit_rate = best_option["hit_rate"]
                n_bets = best_option["n_bets"]
                edge_source = best_option["label"]
                recent_metrics = best_option.get("recent")
            else:
                recent_metrics = None
        elif baseline_reference is not None and baseline_reference.notna().any():
            def _evaluate_numeric_edges(
                preds_array: np.ndarray, label: str
            ) -> Optional[Dict[str, Any]]:
                if preds_array.size != len(out_df) or not len(out_df):
                    return None

                pred_series = pd.Series(preds_array, index=out_df.index, dtype=float)
                baseline_series = baseline_reference.astype(float)
                actual_series = pd.to_numeric(out_df.get(target), errors="coerce")

                mask = actual_series.notna() & baseline_series.notna()
                if "is_synthetic" in out_df.columns:
                    mask &= ~out_df["is_synthetic"].astype(bool)

                if not mask.any():
                    return None

                pred_series = pred_series.loc[mask]
                baseline_series = baseline_series.loc[mask]
                actual_series = actual_series.loc[mask]

                residuals = (actual_series - pred_series).to_numpy(dtype=float)
                residuals = residuals[np.isfinite(residuals)]
                if residuals.size >= 2:
                    sigma = float(np.std(residuals, ddof=1))
                elif residuals.size == 1:
                    sigma = float(abs(residuals[0]))
                else:
                    sigma = float("nan")

                if not math.isfinite(sigma) or sigma < 1e-6:
                    fallback_vals = actual_series.to_numpy(dtype=float)
                    fallback_vals = fallback_vals[np.isfinite(fallback_vals)]
                    if fallback_vals.size >= 2:
                        sigma = float(np.std(fallback_vals, ddof=1))
                    elif fallback_vals.size:
                        sigma = float(max(abs(fallback_vals[0]), 1.0))
                    else:
                        sigma = 1.0

                sigma = max(sigma, 1.0)

                def _normal_cdf(z: float) -> float:
                    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

                payout = _american_to_payout(-110.0)

                def _build_numeric(threshold: float) -> Optional[pd.DataFrame]:
                    rows: List[pd.Series] = []
                    for idx in pred_series.index:
                        mu = pred_series.at[idx]
                        line_val = baseline_series.at[idx]
                        actual_val = actual_series.at[idx]
                        if not (math.isfinite(mu) and math.isfinite(line_val) and math.isfinite(actual_val)):
                            continue

                        z = (line_val - mu) / sigma
                        prob_over = 1.0 - _normal_cdf(z)
                        prob_over = float(np.clip(prob_over, 1e-6, 1 - 1e-6))
                        prob_under = float(np.clip(1.0 - prob_over, 1e-6, 1 - 1e-6))

                        ev_over = ev_of_bet(prob_over, -110.0)
                        ev_under = ev_of_bet(prob_under, -110.0)

                        if ev_over > threshold:
                            series = pd.Series(
                                {
                                    "p_model": prob_over,
                                    "side_win": 1 if actual_val > line_val else 0,
                                    "payout": payout,
                                    "_edge": ev_over,
                                    "_threshold": threshold,
                                    "_side": "over",
                                    "_line": line_val,
                                    "_actual": actual_val,
                                    "_is_push": bool(np.isclose(actual_val, line_val)),
                                }
                            )
                            if np.isclose(actual_val, line_val):
                                series["side_win"] = 0
                            rows.append(series)

                        if ev_under > threshold:
                            series = pd.Series(
                                {
                                    "p_model": prob_under,
                                    "side_win": 1 if actual_val < line_val else 0,
                                    "payout": payout,
                                    "_edge": ev_under,
                                    "_threshold": threshold,
                                    "_side": "under",
                                    "_line": line_val,
                                    "_actual": actual_val,
                                    "_is_push": bool(np.isclose(actual_val, line_val)),
                                }
                            )
                            if np.isclose(actual_val, line_val):
                                series["side_win"] = 0
                            rows.append(series)

                    if not rows:
                        return None
                    return pd.DataFrame(rows)

                candidate_thresholds: List[float] = []
                seen_thresholds: Set[float] = set()
                for candidate in [min_edge, (min_edge * 0.5 if min_edge > 0 else None), 0.0, -0.005]:
                    if candidate is None:
                        continue
                    candidate = float(candidate)
                    if math.isnan(candidate):
                        continue
                    if candidate not in seen_thresholds:
                        candidate_thresholds.append(candidate)
                        seen_thresholds.add(candidate)

                selected_threshold: Optional[float] = None
                picks_df: Optional[pd.DataFrame] = None
                for threshold in candidate_thresholds:
                    candidate_df = _build_numeric(threshold)
                    if candidate_df is not None and not candidate_df.empty:
                        picks_df = candidate_df
                        selected_threshold = threshold
                        break

                if picks_df is None:
                    fallback_df = _build_numeric(float("-inf"))
                    if fallback_df is not None and not fallback_df.empty:
                        fallback_df = fallback_df.sort_values("_edge", ascending=False)
                        top_n = max(1, min(5, len(fallback_df)))
                        picks_df = fallback_df.head(top_n).reset_index(drop=True)
                        selected_threshold = float("-inf")
                        logging.debug("%s: forcing %d numeric bets via fallback selection", target, len(picks_df))

                if picks_df is None:
                    return None

                picks_df["_is_push"] = picks_df["_is_push"].astype(bool)

                label_suffix = label
                if selected_threshold is not None:
                    if selected_threshold == float("-inf"):
                        label_suffix = f"{label_suffix}|forced_top_edges"
                    elif selected_threshold != min_edge:
                        label_suffix = f"{label_suffix}|thr={selected_threshold:+.3f}"
                label_suffix = f"{label_suffix}|priors_baseline"
                bet_count = len(picks_df)
                returns = []
                for _, row in picks_df.iterrows():
                    if row["_is_push"]:
                        returns.append(0.0)
                    elif row["side_win"] == 1:
                        returns.append(row["payout"])
                    else:
                        returns.append(-1.0)
                returns_arr = np.asarray(returns, dtype=float)
                roi_val = float(np.mean(returns_arr))
                m = returns_arr.mean()
                s = returns_arr.std(ddof=1) if bet_count > 1 else 0.0
                se = s / math.sqrt(max(bet_count, 1))
                z = 1.96
                roi_lo_val, roi_hi_val = float(m - z * se), float(m + z * se)
                non_push_mask = ~picks_df["_is_push"].astype(bool)
                if non_push_mask.any():
                    hit_val = float(np.mean(picks_df.loc[non_push_mask, "side_win"]))
                else:
                    hit_val = float("nan")

                return {
                    "roi": roi_val,
                    "roi_lo": roi_lo_val,
                    "roi_hi": roi_hi_val,
                    "hit_rate": hit_val,
                    "n_bets": bet_count,
                    "label": label_suffix,
                }

            prediction_candidates: List[Tuple[str, np.ndarray]] = []
            if len(out_df):
                base_label = "calibrated" if calibration_record is not None else "model"
                prediction_candidates.append((base_label, out_df["_y_pred"].to_numpy(dtype=float)))
                if "_y_pred_raw" in out_df.columns:
                    prediction_candidates.append(("raw", out_df["_y_pred_raw"].to_numpy(dtype=float)))

            best_option: Optional[Dict[str, Any]] = None
            seen_labels: Set[str] = set()
            for label, preds_array in prediction_candidates:
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                evaluation = _evaluate_numeric_edges(preds_array, label)
                if evaluation is None:
                    continue
                if best_option is None or evaluation["roi"] > best_option["roi"]:
                    best_option = evaluation

            if best_option is not None:
                roi = best_option["roi"]
                roi_lo = best_option["roi_lo"]
                roi_hi = best_option["roi_hi"]
                hit_rate = best_option["hit_rate"]
                n_bets = best_option["n_bets"]
                edge_source = best_option["label"]
                recent_metrics = best_option.get("recent")

        # Persist backtest metrics per fold + overall
        for row in per_fold:
            self.db.record_backtest_metrics(self.run_id, f"{target}_fold{row['fold']}", row, sample_size=row["n"])

        evaluation_mode = "walk-forward" if used_walk_forward else "holdout"

        if per_fold:
            summary = {
                "folds": len(per_fold),
                "r2_mean": float(np.mean([r["r2"] for r in per_fold])),
                "mae_mean": float(np.mean([r["mae"] for r in per_fold])),
                "rmse_mean": float(np.mean([r["rmse"] for r in per_fold])),
                "mae_vs_closing": mae_vs_closing,
                "roi": roi,
                "roi_ci": (roi_lo, roi_hi) if roi_lo is not None else None,
                "hit_rate": hit_rate,
                "bets": n_bets,
                "edge_source": edge_source,
                "mode": evaluation_mode,
                "recent": recent_metrics,
            }
        else:
            summary = {
                "folds": 0,
                "r2_mean": float("nan"),
                "mae_mean": float("nan"),
                "rmse_mean": float("nan"),
                "mae_vs_closing": mae_vs_closing,
                "roi": roi,
                "roi_ci": (roi_lo, roi_hi) if roi_lo is not None else None,
                "hit_rate": hit_rate,
                "bets": n_bets,
                "edge_source": edge_source,
                "mode": evaluation_mode,
                "recent": recent_metrics,
            }
        self.db.record_backtest_metrics(self.run_id, target, summary, sample_size=int(len(out_df)))

        recent_log = ""
        if summary.get("recent"):
            recent = summary["recent"]
            recent_roi = recent.get("roi")
            recent_ci = None
            if "roi_lo" in recent and "roi_hi" in recent:
                recent_ci = (recent["roi_lo"], recent["roi_hi"])
            recent_hit = recent.get("hit_rate")
            recent_n = recent.get("n_bets")
            recent_window_start = recent.get("window_start")
            recent_window_end = recent.get("window_end")
            recent_log = " | recent_roi="
            if recent_roi is not None and math.isfinite(recent_roi):
                recent_log += f"{recent_roi:.3f}"
            else:
                recent_log += "n/a"
            if recent_ci and all(math.isfinite(x) for x in recent_ci):
                recent_log += f" (CI=({recent_ci[0]:.3f},{recent_ci[1]:.3f}))"
            if recent_hit is not None and math.isfinite(recent_hit):
                recent_log += f" | recent_hit={recent_hit:.3f}"
            if recent_n is not None:
                recent_log += f" | recent_bets={recent_n}"
            if recent_window_start is not None and recent_window_end is not None:
                recent_log += (
                    " | window="
                    f"{recent_window_start:%Y-%m-%d}->{recent_window_end:%Y-%m-%d}"
                )

        logging.info(
            "%s %s | folds=%d | MAE=%.3f | RMSE=%.3f | MAE_vs_close=%s | ROI=%s (CI=%s) | bets=%d | edge_source=%s%s",
            target,
            evaluation_mode,
            summary["folds"],
            summary["mae_mean"],
            summary["rmse_mean"],
            f"{summary['mae_vs_closing']:.3f}" if summary["mae_vs_closing"] is not None else "n/a",
            f"{summary['roi']:.3f}" if summary["roi"] is not None else "n/a",
            f"({summary['roi_ci'][0]:.3f},{summary['roi_ci'][1]:.3f})" if summary["roi_ci"] else "n/a",
            summary["bets"],
            summary.get("edge_source") or "n/a",
            recent_log,
        )

        # ---- Deployment guard: beat the vig with lower CI bound ----
        if summary["roi_ci"] is not None:
            roi_lower = summary["roi_ci"][0]
            if not (roi_lower > vig_threshold):
                raise RuntimeError(
                    f"{target}: ROI lower CI {roi_lower:.3f} does not beat vig threshold {vig_threshold:.3f}. Aborting deployment."
                )

        special_entry = self.special_models.get(target, {}).copy()
        if calibration_record is not None:
            special_entry["calibration"] = calibration_record

        # Finally fit on ALL data for live predictions
        fit_kwargs = {"regressor__sample_weight": w_all} if w_all is not None else {}
        model.fit(X_all, y_all, **fit_kwargs)
        setattr(model, "feature_columns", list(feature_columns))
        setattr(model, "allowed_positions", TARGET_ALLOWED_POSITIONS.get(target))
        setattr(model, "target_name", target)
        self.feature_column_map[target] = list(feature_columns)
        self.model_uncertainty[target] = {"rmse": summary["rmse_mean"], "mae": summary["mae_mean"]}

        # Supplemental pricing models for prop markets
        fitted_preprocessor = None
        processed_cache: Optional[pd.DataFrame] = None
        processed_feature_names: List[str] = []

        if hasattr(model, "named_steps"):
            fitted_preprocessor = model.named_steps.get("preprocessor")

        def _prepare_processed_frame() -> Optional[pd.DataFrame]:
            nonlocal processed_cache, processed_feature_names
            if processed_cache is not None:
                return processed_cache
            if fitted_preprocessor is None:
                return None
            try:
                transformed = fitted_preprocessor.transform(X_all)
            except Exception:
                logging.debug("Unable to transform features for pricing models on %s", target, exc_info=True)
                return None
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()
            processed_cache = pd.DataFrame(
                transformed,
                index=X_all.index,
                columns=[f"feature_{i}" for i in range(transformed.shape[1])],
            )
            processed_feature_names = list(processed_cache.columns)
            return processed_cache

        if target in {"receiving_yards", "receptions", "passing_yards"}:
            processed = _prepare_processed_frame()
            if processed is not None and len(processed) >= 100:
                try:
                    quant_model = QuantileYards()
                    quant_model.fit(processed, y_all.astype(float))
                    special_entry.update(
                        {
                            "type": "quantile",
                            "model": quant_model,
                            "preprocessor": fitted_preprocessor,
                            "feature_columns": list(feature_columns),
                            "feature_names": processed_feature_names,
                            "allowed_positions": TARGET_ALLOWED_POSITIONS.get(target),
                        }
                    )
                except Exception:
                    logging.exception("Failed to train quantile model for %s", target, exc_info=True)

        if target in {"receiving_tds", "rushing_tds"}:
            processed = _prepare_processed_frame()
            positives = int((y_all > 0).sum())
            negatives = int((y_all <= 0).sum())
            if (
                processed is not None
                and len(processed) >= 150
                and positives >= 5
                and negatives >= 5
            ):
                try:
                    n_splits = max(2, min(5, positives, negatives))
                    hurdle_model = HurdleTDModel()
                    hurdle_model.fit(processed, y_all.astype(float), n_splits=n_splits)
                    special_entry.update(
                        {
                            "type": "hurdle",
                            "model": hurdle_model,
                            "preprocessor": fitted_preprocessor,
                            "feature_columns": list(feature_columns),
                            "feature_names": processed_feature_names,
                            "allowed_positions": TARGET_ALLOWED_POSITIONS.get(target),
                        }
                    )
                except Exception:
                    logging.exception("Failed to train hurdle TD model for %s", target, exc_info=True)

        if special_entry:
            self.special_models[target] = special_entry

        return model, summary

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
            "implied_prob_logit_diff",
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
            "offense_pass_rating_diff",
            "offense_rush_rating_diff",
            "defense_pass_rating_diff",
            "defense_rush_rating_diff",
            "offense_epa_diff",
            "defense_epa_diff",
            "offense_success_rate_diff",
            "defense_success_rate_diff",
            "pace_seconds_diff",
            "travel_penalty_diff",
            "rest_penalty_diff",
            "weather_adjustment_diff",
            "timezone_diff_diff",
            "points_for_avg_diff",
            "points_against_avg_diff",
            "point_diff_avg_diff",
            "win_pct_recent_diff",
            "rest_days_diff",
            "prev_points_for_diff",
            "prev_points_against_diff",
            "prev_point_diff_diff",
            "injury_total_diff",
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

        train_df, test_df, sorted_df = self._chronological_split(df)

        if available_numeric:
            train_numeric = [
                col for col in available_numeric if train_df[col].notna().any()
            ]
            dropped_train_numeric = sorted(set(available_numeric) - set(train_numeric))
            if dropped_train_numeric:
                logging.debug(
                    "Dropping numeric game features with no observed training values: %s",
                    ", ".join(dropped_train_numeric),
                )
            available_numeric = train_numeric

        if not available_numeric and not available_categorical:
            logging.warning(
                "No usable features with observed training values available to train game-level models.",
            )
            return {}

        feature_columns = available_numeric + available_categorical

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

        winner_margin = np.abs(winner_proba - 0.5)
        confidence_profile = self._build_confidence_profile(
            winner_margin, (winner_pred == y_winner_test).astype(int)
        )
        if confidence_profile:
            entry = self.special_models.get("game_winner", {}).copy()
            entry["confidence_profile"] = confidence_profile
            self.special_models["game_winner"] = entry
            for bucket in confidence_profile.get("buckets", []):
                logging.info(
                    "Winner confidence bucket %-6s | margin >= %.3f < %.3f | accuracy=%s | n=%d",
                    bucket.get("label"),
                    bucket.get("lower", float("nan")),
                    bucket.get("upper", float("nan")),
                    f"{bucket.get('accuracy'):.3f}" if bucket.get("accuracy") is not None else "nan",
                    bucket.get("count", 0),
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

        home_error_profile = self._build_error_profile(np.abs(home_pred - y_home_test))
        if home_error_profile:
            entry = self.special_models.get("home_points", {}).copy()
            entry["error_profile"] = home_error_profile
            self.special_models["home_points"] = entry
            logging.info(
                "Home score error profile | median=%.2f | q80=%.2f | q95=%.2f",
                home_error_profile.get("median", float("nan")),
                home_error_profile.get("q80", float("nan")),
                home_error_profile.get("q95", float("nan")),
            )

        away_error_profile = self._build_error_profile(np.abs(away_pred - y_away_test))
        if away_error_profile:
            entry = self.special_models.get("away_points", {}).copy()
            entry["error_profile"] = away_error_profile
            self.special_models["away_points"] = entry
            logging.info(
                "Away score error profile | median=%.2f | q80=%.2f | q95=%.2f",
                away_error_profile.get("median", float("nan")),
                away_error_profile.get("q80", float("nan")),
                away_error_profile.get("q95", float("nan")),
            )

        # Supplemental Poisson totals model for pricing
        try:
            poisson_pre = clone(preprocessor)
            poisson_pre.fit(sorted_df[feature_columns])

            transformed_train = poisson_pre.transform(X_train)
            transformed_test = poisson_pre.transform(X_test)
            transformed_full = poisson_pre.transform(sorted_df[feature_columns])

            if hasattr(transformed_train, "toarray"):
                transformed_train = transformed_train.toarray()
            if hasattr(transformed_test, "toarray"):
                transformed_test = transformed_test.toarray()
            if hasattr(transformed_full, "toarray"):
                transformed_full = transformed_full.toarray()

            try:
                feature_names_out = list(poisson_pre.get_feature_names_out())
            except Exception:
                feature_names_out = []

            expected_cols = transformed_train.shape[1]
            if not feature_names_out or len(feature_names_out) != expected_cols:
                logging.debug(
                    "Poisson preprocessor feature name mismatch: expected %d, got %d; "
                    "using generic feature names.",
                    expected_cols,
                    len(feature_names_out),
                )
                feature_names_out = [f"feature_{i}" for i in range(expected_cols)]

            train_processed = pd.DataFrame(
                transformed_train,
                columns=feature_names_out,
                index=X_train.index,
            )
            test_processed = pd.DataFrame(
                transformed_test,
                columns=feature_names_out,
                index=X_test.index,
            )
            full_processed = pd.DataFrame(
                transformed_full,
                columns=feature_names_out,
                index=sorted_df.index,
            )

            poisson_model = TeamPoissonTotals(alpha=1.0)
            poisson_model.fit(train_processed, y_home_train, train_processed, y_away_train)
            lam_home, lam_away = poisson_model.predict_lambda(test_processed, test_processed)
            rmse_home_pois = compute_rmse(y_home_test, lam_home)
            rmse_away_pois = compute_rmse(y_away_test, lam_away)
            logging.info(
                "Poisson team totals holdout | Home RMSE=%.2f Away RMSE=%.2f",
                rmse_home_pois,
                rmse_away_pois,
            )

            poisson_model.fit(
                full_processed,
                sorted_df["home_score"],
                full_processed,
                sorted_df["away_score"],
            )
            self.special_models["team_poisson"] = {
                "type": "poisson",
                "model": poisson_model,
                "preprocessor": poisson_pre,
                "feature_columns": list(feature_columns),
                "feature_names": feature_names_out,
            }
        except Exception:
            logging.exception("Failed to fit Poisson totals model", exc_info=True)

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
    if trainer is not None and getattr(trainer, "feature_builder", None) is not None:
        feature_builder = trainer.feature_builder
    else:
        supplemental_loader = None
        if trainer is not None:
            supplemental_loader = getattr(trainer, "supplemental_loader", None)
        feature_builder = FeatureBuilder(engine, supplemental_loader)
        feature_builder.build_features()
    if feature_builder.games_frame is None:
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

    now_utc = dt.datetime.now(dt.timezone.utc)
    lookback = now_utc - pd.Timedelta(hours=12)
    lookahead = now_utc + pd.Timedelta(days=7, hours=12)
    eastern_zone = ZoneInfo("America/New_York")
    fallback_schedule_cache: Optional[pd.DataFrame] = None
    fallback_attempted = False
    future_only_cutoff = now_utc - dt.timedelta(minutes=5)

    def _to_utc_timestamp(value: Any) -> pd.Timestamp:
        """Return a timezone-aware UTC pandas Timestamp."""

        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    def _resolve_schedule_start(schedule: Dict[str, Any]) -> Optional[pd.Timestamp]:
        """Resolve a kickoff time from a MySportsFeeds schedule payload."""

        def _parse_timestamp(value: Any) -> Optional[pd.Timestamp]:
            if value is None:
                return None
            if isinstance(value, float) and np.isnan(value):  # type: ignore[arg-type]
                return None
            if isinstance(value, (pd.Timestamp, dt.datetime)):
                ts = pd.Timestamp(value)
                return ts.tz_convert(dt.timezone.utc) if ts.tzinfo else ts.tz_localize(dt.timezone.utc)
            if isinstance(value, dt.date):
                return pd.Timestamp(value)
            if isinstance(value, dict):
                for candidate in (
                    "utc",
                    "iso",
                    "ISO",
                    "dateTime",
                    "date_time",
                    "date",
                    "full",
                ):
                    if candidate in value:
                        ts = _parse_timestamp(value.get(candidate))
                        if ts is not None:
                            return ts
                for nested_value in value.values():
                    ts = _parse_timestamp(nested_value)
                    if ts is not None:
                        return ts
                return None
            try:
                ts = pd.to_datetime(value, utc=False, errors="coerce")
            except Exception:
                return None
            if pd.isna(ts):
                return None
            return ts

        def _parse_time(value: Any) -> Optional[dt.time]:
            if value is None:
                return None
            if isinstance(value, dt.time):
                return value
            if isinstance(value, (pd.Timestamp, dt.datetime)):
                ts = pd.Timestamp(value)
                ts = ts.tz_convert(eastern_zone) if ts.tzinfo else ts
                return ts.time()
            text = str(value).strip()
            if not text or text.lower() in {"tbd", "tba", "na", "none"}:
                return None
            try:
                parsed = pd.to_datetime(text, errors="coerce")
            except Exception:
                return None
            if pd.isna(parsed):
                return None
            return pd.Timestamp(parsed).time()

        primary_fields = (
            "startTimeUTC",
            "startTimeUtc",
            "startTime",
            "originalStartTime",
            "startDateTime",
            "startTimeISO",
            "startTimeIso",
            "dateTime",
        )
        for field in primary_fields:
            ts = _parse_timestamp(schedule.get(field))
            if ts is None:
                continue
            if ts.tzinfo is None:
                ts = ts.tz_localize(dt.timezone.utc)
            else:
                ts = ts.tz_convert(dt.timezone.utc)
            return ts

        date_fields = (
            "startDate",
            "originalStartDate",
            "date",
            "gameDate",
            "day",
        )
        time_fields = (
            "startTime",
            "startTimeLocal",
            "startTimeEastern",
            "startTimeET",
            "gameTime",
            "time",
            "localStartTime",
        )

        for field in date_fields:
            ts = _parse_timestamp(schedule.get(field))
            if ts is None:
                continue

            if ts.tzinfo is None:
                base_date = ts.date()
            else:
                base_date = ts.astimezone(eastern_zone).date()

            kickoff_time: Optional[dt.time] = None
            for time_field in time_fields:
                kickoff_time = _parse_time(schedule.get(time_field))
                if kickoff_time is not None:
                    break

            if kickoff_time is None:
                kickoff_time = dt.time(hour=13, minute=0)

            localized = dt.datetime.combine(base_date, kickoff_time, tzinfo=eastern_zone)
            return pd.Timestamp(localized.astimezone(dt.timezone.utc))

        return None

    def _fetch_msf_schedule(reason: str) -> pd.DataFrame:
        nonlocal fallback_schedule_cache, fallback_attempted

        if fallback_schedule_cache is not None:
            return fallback_schedule_cache
        if fallback_attempted:
            return pd.DataFrame()

        fallback_attempted = True

        msf_client = getattr(ingestor, "msf_client", None) if ingestor is not None else None
        if msf_client is None:
            logging.debug(
                "Cannot fetch MySportsFeeds schedule fallback (%s): ingestor has no msf_client",
                reason,
            )
            return pd.DataFrame()

        if config is not None and getattr(config, "seasons", None):
            seasons_to_query = [str(season) for season in config.seasons if season]
        elif "season" in games_source.columns:
            seasons_to_query = (
                pd.Series(games_source["season"])
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        else:
            seasons_to_query = []

        seasons_to_query = [season for season in seasons_to_query if season]
        if not seasons_to_query:
            seasons_to_query = [f"{now_utc.year}-regular"]

        fallback_rows: List[Dict[str, Any]] = []
        seen_game_ids: Set[str] = set()

        def _append_msf_game(
            game: Dict[str, Any], season_key: str
        ) -> Optional[pd.Timestamp]:
            schedule = game.get("schedule") or {}
            game_id = schedule.get("id")
            if not game_id:
                return None

            game_id_str = str(game_id)
            if game_id_str in seen_game_ids:
                return None

            start_time_ts = _resolve_schedule_start(schedule)
            if start_time_ts is None:
                return None

            if start_time_ts.tzinfo is None:
                start_time_ts = start_time_ts.tz_localize(dt.timezone.utc)

            if start_time_ts < _to_utc_timestamp(future_only_cutoff):
                return None

            home_info = schedule.get("homeTeam") or {}
            away_info = schedule.get("awayTeam") or {}
            home_team = normalize_team_abbr(
                home_info.get("abbreviation") or home_info.get("name")
            )
            away_team = normalize_team_abbr(
                away_info.get("abbreviation") or away_info.get("name")
            )
            if not home_team or not away_team:
                return None

            week_value = schedule.get("week")
            try:
                week_int = int(week_value) if week_value is not None else None
            except (TypeError, ValueError):
                week_int = None

            status_raw = schedule.get("status") or schedule.get("playedStatus")
            status = str(status_raw).lower() if status_raw else "scheduled"

            venue_info = schedule.get("venue")
            if isinstance(venue_info, dict):
                venue_value = (
                    venue_info.get("name")
                    or venue_info.get("venueName")
                    or venue_info.get("stadium")
                    or ""
                )
            else:
                venue_value = venue_info or ""

            referee_info = schedule.get("referee")
            if isinstance(referee_info, dict):
                referee_value = referee_info.get("name") or ""
            else:
                referee_value = referee_info or ""

            fallback_rows.append(
                {
                    "game_id": game_id_str,
                    "season": str(season_key),
                    "week": week_int,
                    "start_time": start_time_ts.to_pydatetime(),
                    "day_of_week": start_time_ts.tz_convert(eastern_zone).strftime("%A"),
                    "home_team": home_team,
                    "away_team": away_team,
                    "status": status,
                    "venue": venue_value,
                    "referee": referee_value,
                    "home_score": np.nan,
                    "away_score": np.nan,
                    "odds_updated": pd.NaT,
                }
            )
            seen_game_ids.add(game_id_str)
            return start_time_ts

        for season_key in seasons_to_query:
            range_start = (now_utc - dt.timedelta(days=1)).date()
            range_end = (now_utc + dt.timedelta(days=14)).date()
            status_candidates: Tuple[Optional[str], ...] = (
                "scheduled",
                "upcoming",
                "unplayed",
                "pre",
                None,
            )

            season_has_future = False

            for status_filter in status_candidates:
                try:
                    date_range_games = msf_client.fetch_games_by_date_range(
                        season_key,
                        range_start,
                        range_end,
                        status=status_filter,
                    )
                except Exception:
                    logging.warning(
                        "Failed to fetch date-range schedule from MySportsFeeds for %s",
                        season_key,
                        exc_info=True,
                    )
                    continue

                if not date_range_games:
                    continue

                logging.debug(
                    "Loaded %d scheduled games for %s via date-range status '%s'",
                    len(date_range_games),
                    season_key,
                    status_filter or "any",
                )

                for game in date_range_games:
                    appended_ts = _append_msf_game(game, season_key)
                    if appended_ts is not None and appended_ts >= _to_utc_timestamp(now_utc):
                        season_has_future = True

                if season_has_future:
                    break

            if not season_has_future:
                try:
                    season_games = msf_client.fetch_games(season_key)
                except Exception:
                    logging.warning(
                        "Failed to fetch full season schedule from MySportsFeeds for %s",
                        season_key,
                        exc_info=True,
                    )
                    continue

                for game in season_games or []:
                    appended_ts = _append_msf_game(game, season_key)
                    if appended_ts is not None and appended_ts >= _to_utc_timestamp(now_utc):
                        season_has_future = True

                if not season_has_future:
                    logging.debug(
                        "No future games detected for %s between %s and %s using MySportsFeeds data",
                        season_key,
                        range_start,
                        range_end,
                    )

        if not fallback_rows:
            fallback_schedule_cache = pd.DataFrame()
        else:
            fallback_schedule_cache = pd.DataFrame(fallback_rows)
            fallback_schedule_cache = fallback_schedule_cache.drop_duplicates(
                subset=["game_id"], keep="last"
            )
            fallback_schedule_cache["start_time"] = pd.to_datetime(
                fallback_schedule_cache["start_time"], utc=True, errors="coerce"
            )
            fallback_schedule_cache = fallback_schedule_cache[
                fallback_schedule_cache["start_time"].notna()
            ]

            future_cutoff = _to_utc_timestamp(future_only_cutoff)
            fallback_schedule_cache = fallback_schedule_cache[
                fallback_schedule_cache["start_time"] >= future_cutoff
            ]

            fallback_schedule_cache["day_of_week"] = fallback_schedule_cache["day_of_week"].where(
                fallback_schedule_cache["day_of_week"].notna(),
                fallback_schedule_cache["start_time"].dt.day_name(),
            )

            if fallback_schedule_cache.empty:
                logging.warning(
                    "MySportsFeeds fallback (%s) did not return any games with kickoffs after %s",
                    reason,
                    future_cutoff.tz_convert(eastern_zone),
                )
            else:
                logging.info(
                    "Loaded %d upcoming games from MySportsFeeds fallback (%s)",
                    len(fallback_schedule_cache),
                    reason,
                )

        return fallback_schedule_cache.copy()

    def _prepare_schedule_frame(frame: pd.DataFrame, source_label: str) -> pd.DataFrame:
        if frame.empty:
            return frame

        working = frame.copy()
        if "home_team" in working.columns:
            working["home_team"] = working["home_team"].apply(normalize_team_abbr)
        if "away_team" in working.columns:
            working["away_team"] = working["away_team"].apply(normalize_team_abbr)

        required_cols = {"home_team", "away_team"}
        if required_cols.issubset(working.columns):
            working = working[working["home_team"].notna() & working["away_team"].notna()]

        if working.empty:
            return working

        working["start_time"] = pd.to_datetime(working.get("start_time"), utc=True, errors="coerce")
        working = working[working["start_time"].notna()]
        if working.empty:
            return working

        working["local_start_time"] = working["start_time"].dt.tz_convert(eastern_zone)
        working["local_day_of_week"] = working["local_start_time"].dt.day_name()
        if "day_of_week" in working.columns:
            working["day_of_week"] = working["day_of_week"].where(
                working["day_of_week"].notna(), working["local_day_of_week"]
            )
        else:
            working["day_of_week"] = working["local_day_of_week"]

        if "season" in working.columns:
            working["season"] = working["season"].astype(str)

        text_normalize_columns = {"venue", "referee", "stadium", "venue_name"}
        for text_col in text_normalize_columns:
            if text_col in working.columns:
                series = working[text_col]
                series = series.where(series.notna(), "")
                working[text_col] = series.astype(str).str.strip()

        if "venue" not in working.columns and "venue_name" in working.columns:
            working["venue"] = working["venue_name"]

        status_series = working.get("status")
        if status_series is not None:
            working["status"] = status_series.fillna("scheduled").astype(str).str.lower()
        else:
            working["status"] = "scheduled"

        if "odds_updated" not in working.columns:
            working["odds_updated"] = pd.NaT

        working["_schedule_source"] = source_label

        return working

    def _select_upcoming(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame

        status_whitelist = {
            "upcoming",
            "scheduled",
            "inprogress",
            "pre",
            "pregame",
            "pre-game",
            "tbd",
            "unplayed",
        }

        schedule = frame[
            frame["status"].isin(status_whitelist)
            | frame["home_score"].isna()
            | frame["away_score"].isna()
        ].copy()
        if schedule.empty:
            logging.warning("No upcoming games found for prediction")
            return pd.DataFrame()

        schedule = schedule.sort_values("start_time").reset_index(drop=True)

        future_mask = schedule["start_time"] >= now_utc
        upcoming = schedule.loc[future_mask].copy()
        if upcoming.empty:
            upcoming = schedule[schedule["start_time"] >= lookback].copy()

        if upcoming.empty:
            logging.warning(
                "Upcoming schedule only contains games more than 12 hours in the past"
            )
            return pd.DataFrame()

        window_mask = upcoming["start_time"] <= lookahead
        window_games = upcoming.loc[window_mask].copy()

        if window_games.empty:
            earliest_start = upcoming["start_time"].min()
            if pd.isna(earliest_start):
                logging.warning("No upcoming games have a valid kickoff time available")
                return pd.DataFrame()

            week_start = earliest_start.normalize() - pd.to_timedelta(
                earliest_start.weekday(), unit="D"
            )
            week_end = week_start + pd.Timedelta(days=7)
            week_mask = (upcoming["start_time"] >= week_start) & (
                upcoming["start_time"] <= week_end
            )
            window_games = upcoming.loc[week_mask].copy()
            if window_games.empty:
                logging.warning("No upcoming games within the fallback week window")
                return pd.DataFrame()

            logging.info(
                "Falling back to future scheduled week %s-%s with %d games",
                week_start.date(),
                week_end.date(),
                len(window_games),
            )

        selection = window_games.copy()

        desired_days = {"Thursday", "Saturday", "Sunday", "Monday"}
        day_mask = selection["local_day_of_week"].isin(desired_days)
        if day_mask.any():
            selection = selection.loc[day_mask].copy()
        else:
            logging.warning(
                "No Thursday/Saturday/Sunday/Monday games available for prediction; using full schedule window"
            )

        selection.loc[:, "_priority"] = selection["_schedule_source"].map(
            {"MySportsFeeds schedule": 0, "database schedule": 1}
        ).fillna(1)
        selection = selection.sort_values(
            ["_priority", "odds_updated", "start_time"], ascending=[True, False, True]
        )
        selection = selection.drop_duplicates(
            subset=["home_team", "away_team", "start_time"], keep="first"
        )
        selection = selection.drop(columns="_priority", errors="ignore")

        selection = selection.sort_values("start_time").reset_index(drop=True)

        for text_col in ("venue", "referee"):
            if text_col in selection.columns:
                series = selection[text_col]
                series = series.where(series.notna(), "")
                selection[text_col] = series.astype(str).str.strip()

        return selection

    normalized_games = _prepare_schedule_frame(games_source, "database schedule")

    fallback_frame = _fetch_msf_schedule("primary upcoming schedule refresh")
    fallback_normalized = pd.DataFrame()
    schedule_frames: List[pd.DataFrame] = []

    if not fallback_frame.empty:
        fallback_normalized = _prepare_schedule_frame(
            fallback_frame, "MySportsFeeds schedule"
        )
        if not fallback_normalized.empty:
            schedule_frames.append(fallback_normalized)

    if not normalized_games.empty:
        schedule_frames.append(normalized_games)

    combined_schedule = safe_concat(schedule_frames)

    if combined_schedule.empty:
        logging.warning("No upcoming schedule data available from MySportsFeeds or database")
        return {"games": pd.DataFrame(), "players": pd.DataFrame()}

    upcoming = pd.DataFrame()
    if not fallback_normalized.empty:
        upcoming = _select_upcoming(fallback_normalized)
        if upcoming.empty:
            logging.warning(
                "MySportsFeeds schedule did not yield upcoming games after filtering; "
                "falling back to combined schedule"
            )

    if upcoming.empty:
        upcoming = _select_upcoming(combined_schedule)

    if upcoming.empty:
        logging.warning("No upcoming games found for prediction")
        return {"games": pd.DataFrame(), "players": pd.DataFrame()}

    def _ensure_model_features(frame: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
        columns: Optional[Iterable[str]] = getattr(model, "feature_columns", None)
        if not columns:
            target_key = getattr(model, "target_name", None)
            if trainer is not None and target_key:
                columns = trainer.feature_column_map.get(target_key)
        if not columns:
            columns = getattr(model, "feature_names_in_", None)

        if not columns:
            numeric_cols = frame.select_dtypes(include=[np.number, bool]).columns.tolist()
            if not numeric_cols:
                return frame
            return frame.loc[:, numeric_cols]

        column_list = list(columns)
        if not column_list:
            numeric_cols = frame.select_dtypes(include=[np.number, bool]).columns.tolist()
            if not numeric_cols:
                return frame
            return frame.loc[:, numeric_cols]

        frame = frame.copy()
        for col in column_list:
            if col not in frame.columns:
                frame[col] = np.nan

        return frame.reindex(columns=column_list)

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

    specials: Dict[str, Any] = {}
    if trainer is not None:
        try:
            specials = getattr(trainer, "special_models", {}) or {}
        except Exception:
            specials = {}
        calibration_info = specials.get("game_winner", {}).get("calibration") if isinstance(specials, dict) else None
        if calibration_info and calibration_info.get("method") == "logistic":
            model = calibration_info.get("model")
            if model is not None:
                try:
                    winner_probs = model.predict_proba(winner_probs.reshape(-1, 1))[:, 1]
                except Exception:
                    logging.debug(
                        "Unable to apply stored calibration for game_winner during inference",
                        exc_info=True,
                    )

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
    scoreboard["home_win_margin"] = (scoreboard["home_win_probability"] - 0.5).abs()
    scoreboard["home_win_confidence"] = pd.Series(pd.NA, index=scoreboard.index, dtype="object")
    scoreboard["home_win_confidence_accuracy"] = np.nan
    scoreboard["home_win_edge"] = np.nan
    scoreboard["bet_recommendation"] = "monitor"
    scoreboard["high_confidence_flag"] = False
    scoreboard["home_score_expected_error"] = np.nan
    scoreboard["away_score_expected_error"] = np.nan

    confidence_profile = {}
    if isinstance(specials, dict):
        confidence_profile = specials.get("game_winner", {}).get("confidence_profile", {})
    if confidence_profile:
        thresholds = confidence_profile.get("thresholds")
        bucket_records = confidence_profile.get("buckets", [])
        bucket_labels = [record.get("label") for record in bucket_records if record.get("label")]
        if thresholds and bucket_labels:
            try:
                confidence_series = pd.cut(
                    scoreboard["home_win_margin"],
                    bins=thresholds,
                    labels=bucket_labels,
                    include_lowest=True,
                    right=False,
                )
                scoreboard["home_win_confidence"] = confidence_series.astype(str)
                scoreboard.loc[
                    confidence_series.isna(), "home_win_confidence"
                ] = pd.NA
            except Exception:
                logging.debug("Unable to bucketize win confidence", exc_info=True)
        acc_map = {record.get("label"): record.get("accuracy") for record in bucket_records}
        scoreboard["home_win_confidence_accuracy"] = scoreboard["home_win_confidence"].map(acc_map)
        preferred_label = confidence_profile.get("preferred_label")
        if preferred_label:
            scoreboard["high_confidence_flag"] = scoreboard["home_win_confidence"] == preferred_label

    home_error_profile = (specials.get("home_points", {}) or {}).get("error_profile") if isinstance(specials, dict) else None
    if home_error_profile:
        scoreboard["home_score_expected_error"] = home_error_profile.get("median")

    away_error_profile = (specials.get("away_points", {}) or {}).get("error_profile") if isinstance(specials, dict) else None
    if away_error_profile:
        scoreboard["away_score_expected_error"] = away_error_profile.get("median")

    poisson_entry = specials.get("team_poisson") if isinstance(specials, dict) else None
    if poisson_entry and not game_features.empty:
        base_features = _ensure_model_features(
            game_features,
            SimpleNamespace(feature_columns=poisson_entry.get("feature_columns", [])),
        )
        processed = _transform_with_feature_names(
            poisson_entry.get("preprocessor"),
            base_features,
            poisson_entry.get("feature_names"),
        )
        try:
            lam_home, lam_away = poisson_entry["model"].predict_lambda(processed, processed)
        except Exception:
            logging.debug("Unable to score Poisson consensus for upcoming games", exc_info=True)
            lam_home = lam_away = np.array([])
        if lam_home.size and lam_away.size:
            poisson_probs = [
                TeamPoissonTotals._home_win_probability_pair(float(h), float(a))
                for h, a in zip(lam_home, lam_away)
            ]
            poisson_df = pd.DataFrame(
                {
                    "game_id": game_features["game_id"].values,
                    "home_score_poisson": lam_home,
                    "away_score_poisson": lam_away,
                    "home_win_poisson_prob": poisson_probs,
                }
            )
            scoreboard = scoreboard.merge(poisson_df, on="game_id", how="left")
            scoreboard["home_score_consensus_gap"] = (
                scoreboard["home_score"] - scoreboard["home_score_poisson"]
            )
            scoreboard["away_score_consensus_gap"] = (
                scoreboard["away_score"] - scoreboard["away_score_poisson"]
            )
            scoreboard["home_win_consensus_gap"] = (
                scoreboard["home_win_probability"] - scoreboard["home_win_poisson_prob"]
            )

    def _blend_with_consensus(
        primary: pd.Series,
        consensus: pd.Series,
        conf_series: pd.Series,
        gap_series: pd.Series,
        *,
        base: float = 0.55,
    ) -> pd.Series:
        if consensus is None or consensus.empty:
            return primary
        conf_map = {"high": 0.8, "medium": 0.6, "low": 0.45}
        conf_weights = conf_series.str.lower().map(conf_map).fillna(base)
        gap_penalty = (gap_series.abs().clip(0.0, 0.3) / 0.3) * 0.2
        weights = (conf_weights - gap_penalty).clip(0.35, 0.9)
        blended = weights * primary + (1.0 - weights) * consensus
        return blended

    if {
        "home_score_poisson",
        "away_score_poisson",
        "home_win_poisson_prob",
    }.issubset(scoreboard.columns):
        conf_series = scoreboard["home_win_confidence"].fillna("")
        scoreboard["home_win_probability"] = _blend_with_consensus(
            scoreboard["home_win_probability"],
            scoreboard["home_win_poisson_prob"],
            conf_series,
            scoreboard.get("home_win_consensus_gap", pd.Series(0.0, index=scoreboard.index)),
        )
        scoreboard["home_score"] = _blend_with_consensus(
            scoreboard["home_score"],
            scoreboard["home_score_poisson"],
            conf_series,
            scoreboard.get("home_score_consensus_gap", pd.Series(0.0, index=scoreboard.index)),
            base=0.6,
        )
        scoreboard["away_score"] = _blend_with_consensus(
            scoreboard["away_score"],
            scoreboard["away_score_poisson"],
            conf_series,
            scoreboard.get("away_score_consensus_gap", pd.Series(0.0, index=scoreboard.index)),
            base=0.6,
        )
        scoreboard["home_win_consensus_gap"] = (
            scoreboard["home_win_probability"] - scoreboard["home_win_poisson_prob"]
        )
        scoreboard["home_score_consensus_gap"] = (
            scoreboard["home_score"] - scoreboard["home_score_poisson"]
        )
        scoreboard["away_score_consensus_gap"] = (
            scoreboard["away_score"] - scoreboard["away_score_poisson"]
        )

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

    odds_columns = [
        "home_moneyline",
        "away_moneyline",
        "home_implied_prob",
        "away_implied_prob",
        "odds_updated",
    ]
    available_odds = [col for col in odds_columns if col in game_features.columns]
    if available_odds:
        odds_frame = game_features[["game_id", *available_odds]].drop_duplicates("game_id")
        scoreboard = scoreboard.merge(odds_frame, on="game_id", how="left")

    def _sanitize_probabilities(column: str) -> None:
        if column not in scoreboard.columns:
            return
        probs = pd.to_numeric(scoreboard[column], errors="coerce")
        invalid = ~np.isfinite(probs) | (probs <= 0.0) | (probs >= 1.0)
        scoreboard[column] = probs.mask(invalid)

    _sanitize_probabilities("home_implied_prob")
    _sanitize_probabilities("away_implied_prob")

    if "home_implied_prob" in scoreboard.columns:
        scoreboard["home_win_edge"] = scoreboard["home_win_probability"] - scoreboard["home_implied_prob"]
    if "away_implied_prob" in scoreboard.columns:
        scoreboard["away_win_edge"] = (1.0 - scoreboard["home_win_probability"]) - scoreboard["away_implied_prob"]

    edge_threshold = 0.05
    if confidence_profile:
        edge_threshold = float(confidence_profile.get("edge_threshold", edge_threshold))

    consensus_gap = scoreboard.get("home_win_consensus_gap")
    if consensus_gap is None:
        consensus_gap = pd.Series(0.0, index=scoreboard.index)
    scoreboard["consensus_warning"] = np.where(
        consensus_gap.abs() >= 0.1,
        "model_disagrees",
        "",
    )

    confidence_text = scoreboard["home_win_confidence"].fillna("")
    scoreboard["bet_recommendation"] = np.select(
        [
            scoreboard["high_confidence_flag"]
            & scoreboard["home_win_edge"].abs().ge(edge_threshold),
            scoreboard["high_confidence_flag"],
        ],
        ["target", "lean"],
        default=np.where(confidence_text == "medium", "monitor", "pass"),
    )

    scoreboard = scoreboard.sort_values(["date", "start_time", "game_id"]).reset_index(drop=True)

    base_columns = [
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
        "home_score_expected_error",
        "away_score_expected_error",
        "home_score_poisson",
        "away_score_poisson",
        "home_score_consensus_gap",
        "away_score_consensus_gap",
        "home_win_probability",
        "home_win_poisson_prob",
        "home_win_consensus_gap",
        "home_win_margin",
        "home_win_confidence",
        "home_win_confidence_accuracy",
        "home_win_edge",
        "away_win_edge",
        "home_win_log_loss",
        "home_win_brier",
        "home_win_accuracy",
        "bet_recommendation",
        "consensus_warning",
        "high_confidence_flag",
    ]
    odds_cols_present = [col for col in odds_columns if col in scoreboard.columns]
    final_columns = base_columns + odds_cols_present
    available_final = [col for col in final_columns if col in scoreboard.columns]

    scoreboard = scoreboard[available_final].rename(
        columns={
            "away_team": "away_team_abbr",
            "home_team": "home_team_abbr",
        }
    )

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
        if "_usage_confidence" in player_features.columns:
            player_predictions["_usage_confidence"] = player_features[
                "_usage_confidence"
            ].values
        if "is_placeholder" in player_features.columns:
            player_predictions["is_placeholder"] = (
                player_features["is_placeholder"].astype(bool).values
            )

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
                if trainer is not None:
                    preds = trainer.calibrate_player_predictions(
                        target,
                        player_features.loc[mask],
                        preds,
                    )
                if target in NON_NEGATIVE_TARGETS:
                    preds = np.maximum(np.asarray(preds, dtype=float), 0.0)
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
        player_predictions[value_columns] = (
            player_predictions[value_columns].fillna(0.0).clip(lower=0.0)
        )

        def _apply_position_quantile_brakes(
            df: pd.DataFrame,
            trainer_obj: Optional["ModelTrainer"],
            target_map: Dict[str, Tuple[str, str]],
        ) -> pd.DataFrame:
            if trainer_obj is None:
                return df
            adjusted = df.copy()
            for col, (team_col, pos_col) in target_map.items():
                if col not in adjusted.columns:
                    continue
                values = adjusted[col].astype(float)
                lowers: List[float] = []
                uppers: List[float] = []
                for row in adjusted[[team_col, pos_col]].itertuples(index=False):
                    lower_q, upper_q = trainer_obj._resolve_prior_bounds(
                        col.replace("pred_", ""), row[0], row[1]
                    )
                    lowers.append(lower_q)
                    uppers.append(upper_q)
                lower_series = pd.Series(lowers, index=values.index)
                upper_series = pd.Series(uppers, index=values.index)
                values = values.mask(values < lower_series, lower_series)
                values = values.mask(values > upper_series, upper_series)
                adjusted[col] = values
            return adjusted

        player_predictions = _apply_position_quantile_brakes(
            player_predictions,
            trainer,
            {
                "pred_rushing_yards": ("team", "position"),
                "pred_receiving_yards": ("team", "position"),
                "pred_receptions": ("team", "position"),
                "pred_passing_yards": ("team", "position"),
            },
        )

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

    priced_results: Dict[str, pd.DataFrame] = {}
    if trainer is not None and player_features is not None:
        specials = getattr(trainer, "special_models", {}) or {}
        player_pred_tables: Dict[str, pd.DataFrame] = {}

        def _build_player_index(frame: pd.DataFrame) -> pd.DataFrame:
            cols = [
                col
                for col in [
                    "player_id",
                    "player_name",
                    "team",
                    "opponent",
                    "position",
                    "game_id",
                    "event_id",
                ]
                if col in frame.columns
            ]
            return frame[cols].copy()

        for target in ("receiving_yards", "receptions", "passing_yards"):
            info = specials.get(target)
            if not info or info.get("type") != "quantile":
                continue
            allowed = info.get("allowed_positions")
            mask = (
                player_features["position"].isin(allowed)
                if allowed and "position" in player_features.columns
                else pd.Series(True, index=player_features.index)
            )
            subset = player_features.loc[mask].copy()
            if subset.empty:
                continue
            features_subset = _ensure_model_features(
                subset, SimpleNamespace(feature_columns=info.get("feature_columns", []))
            )
            player_index = _build_player_index(subset)
            table = make_quantile_pred_table(
                info["model"],
                info["preprocessor"],
                info.get("feature_names"),
                features_subset,
                player_index,
                target,
            )
            if not table.empty:
                player_pred_tables[target] = table

        anytime_table = pd.DataFrame()
        for target in ("receiving_tds", "rushing_tds"):
            info = specials.get(target)
            if not info or info.get("type") != "hurdle":
                continue
            allowed = info.get("allowed_positions")
            mask = (
                player_features["position"].isin(allowed)
                if allowed and "position" in player_features.columns
                else pd.Series(True, index=player_features.index)
            )
            subset = player_features.loc[mask].copy()
            if subset.empty:
                continue
            features_subset = _ensure_model_features(
                subset, SimpleNamespace(feature_columns=info.get("feature_columns", []))
            )
            player_index = _build_player_index(subset)
            table = make_anytime_td_table(
                info["model"],
                info["preprocessor"],
                info.get("feature_names"),
                features_subset,
                player_index,
            )
            if table.empty:
                continue
            if anytime_table.empty:
                anytime_table = table
            else:
                key_cols = [col for col in ["player_id", "team", "opponent"] if col in table.columns]
                merged = anytime_table.merge(
                    table,
                    on=key_cols,
                    how="outer",
                    suffixes=("_a", "_b"),
                )
                merged["anytime_prob"] = merged[["anytime_prob_a", "anytime_prob_b"]].mean(axis=1)
                anytime_table = merged.drop(columns=["anytime_prob_a", "anytime_prob_b"], errors="ignore")
        if not anytime_table.empty:
            player_pred_tables["anytime_td"] = anytime_table

        poisson_info = specials.get("team_poisson")
        feats_home = pd.DataFrame()
        feats_away = pd.DataFrame()
        if poisson_info and not game_features.empty:
            base_features = _ensure_model_features(
                game_features,
                SimpleNamespace(feature_columns=poisson_info.get("feature_columns", [])),
            )
            processed = _transform_with_feature_names(
                poisson_info["preprocessor"],
                base_features,
                poisson_info.get("feature_names"),
            )
            feats_home = processed.copy()
            feats_away = processed.copy()

        if player_pred_tables or not feats_home.empty:
            start_min = upcoming["start_time"].min()
            start_max = upcoming["start_time"].max()
            if pd.isna(start_min) or pd.isna(start_max):
                week_key = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
            else:
                start_min_ts = pd.to_datetime(start_min)
                start_max_ts = pd.to_datetime(start_max)
                if start_min_ts.tzinfo is None:
                    start_min_ts = start_min_ts.tz_localize(dt.timezone.utc)
                else:
                    start_min_ts = start_min_ts.tz_convert(dt.timezone.utc)
                if start_max_ts.tzinfo is None:
                    start_max_ts = start_max_ts.tz_localize(dt.timezone.utc)
                else:
                    start_max_ts = start_max_ts.tz_convert(dt.timezone.utc)
                start_min = start_min_ts.tz_localize(None)
                start_max = start_max_ts.tz_localize(None)
                week_key = f"{start_min.strftime('%Y%m%d')}_{start_max.strftime('%Y%m%d')}"

            odds_players_df = pd.DataFrame(
                columns=[
                    "market",
                    "player_id",
                    "player_name",
                    "team",
                    "opponent",
                    "line",
                    "american_odds",
                    "sportsbook",
                    "event_id",
                ]
            )
            odds_games_df = pd.DataFrame(
                columns=[
                    "market",
                    "game_id",
                    "away_team",
                    "home_team",
                    "side",
                    "total",
                    "american_odds",
                    "sportsbook",
                    "event_id",
                    "last_update",
                ]
            )

            if ingestor is not None and getattr(ingestor, "odds_client", None) is not None:
                try:
                    odds_payload = ingestor.odds_client.fetch_odds()
                    odds_players_df, odds_games_df = extract_pricing_odds(
                        odds_payload,
                        valid_game_ids=upcoming["game_id"].astype(str).unique(),
                    )
                    logging.info(
                        "Collected %d totals rows and %d player prop rows for pricing",
                        len(odds_games_df),
                        len(odds_players_df),
                    )
                except Exception as exc:
                    logging.warning("Failed to fetch odds for pricing: %s", exc)

            out_dir = Path("pricing_outputs")
            priced_results = emit_priced_picks(
                week_key=week_key,
                player_pred_tables=player_pred_tables,
                odds_players=odds_players_df,
                week_games_df=upcoming[["game_id", "away_team", "home_team"]].copy(),
                feats_home=feats_home,
                feats_away=feats_away,
                odds_games=odds_games_df,
                tpois=(poisson_info or {}).get("model") if poisson_info else None,
                out_dir=out_dir,
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

    def _fmt_pct(value: Optional[float], digits: int = 1) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "—"
        return f"{float(value) * 100:.{digits}f}%"

    def _fmt_signed_pct(value: Optional[float], digits: int = 1) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "—"
        return f"{float(value) * 100:+.{digits}f}%"

    if not scoreboard.empty:
        scoreboard_rows: List[List[str]] = []
        reliability_rows: List[List[str]] = []

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

            confidence_label = row.home_win_confidence or "—"
            confidence_label = confidence_label.capitalize() if confidence_label not in {pd.NA, None, ""} else "—"
            hist_acc = _fmt_pct(row.home_win_confidence_accuracy, digits=0)
            if confidence_label == "—" and hist_acc == "—":
                confidence_display = "—"
            elif hist_acc == "—":
                confidence_display = confidence_label
            elif confidence_label == "—":
                confidence_display = hist_acc
            else:
                confidence_display = f"{confidence_label} ({hist_acc})"

            edge_text = _fmt_signed_pct(row.home_win_edge)
            consensus_gap = _fmt_signed_pct(getattr(row, "home_win_consensus_gap", np.nan))
            if getattr(row, "consensus_warning", ""):
                warning = str(row.consensus_warning).replace("_", " ")
                consensus_gap = f"{consensus_gap} {warning}" if consensus_gap != "—" else warning

            reliability_rows.append(
                [
                    f"{row.away_team_abbr} @ {row.home_team_abbr}",
                    _fmt_pct(row.home_win_probability),
                    edge_text,
                    confidence_display,
                    consensus_gap,
                    row.bet_recommendation,
                ]
            )

        lines.extend(
            _format_table(
                ["Date", "Away", "Home", "Away Score ±RMSE", "Home Score ±RMSE"],
                scoreboard_rows,
                aligns=["left", "left", "left", "right", "right"],
            )
        )

        if reliability_rows:
            lines.append("")
            lines.extend(
                _format_table(
                    [
                        "Matchup",
                        "Home win%",
                        "Edge vs market",
                        "Confidence (hist)",
                        "Consensus gap",
                        "Action",
                    ],
                    reliability_rows,
                    aligns=["left", "right", "right", "left", "left", "left"],
                )
            )

        targets = scoreboard[scoreboard["bet_recommendation"] == "target"]
        if not targets.empty:
            lines.append("")
            lines.append("High-confidence targets:")
            for game in targets.itertuples(index=False):
                win_pct = _fmt_pct(game.home_win_probability)
                edge_txt = _fmt_signed_pct(game.home_win_edge)
                conf = game.home_win_confidence or "—"
                hist_acc = _fmt_pct(game.home_win_confidence_accuracy, digits=0)
                conf_part = conf.capitalize() if conf not in {pd.NA, None, ""} else "—"
                if hist_acc != "—":
                    conf_part = f"{conf_part} (~{hist_acc})"
                implied = _fmt_pct(getattr(game, "home_implied_prob", np.nan))
                implied_part = f", market {implied}" if implied != "—" else ""
                lines.append(
                    "  "
                    + f"{game.away_team_abbr} @ {game.home_team_abbr}: {win_pct} home win prob, "
                    + f"edge {edge_txt}{implied_part}, confidence {conf_part}"
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
                def _non_negative(val: float) -> float:
                    try:
                        numeric = float(val)
                    except (TypeError, ValueError):
                        return 0.0
                    if math.isnan(numeric) or math.isinf(numeric):
                        return 0.0
                    return numeric if numeric >= 0.0 else 0.0

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
                            f"{_non_negative(player.pred_passing_yards):.2f}",
                            f"{_non_negative(player.pred_rushing_yards):.2f}",
                            f"{_non_negative(player.pred_receiving_yards):.2f}",
                            f"{_non_negative(player.pred_receptions):.2f}",
                            f"{_non_negative(display_touchdowns):.2f}",
                            f"{_non_negative(player.pred_passing_tds):.2f}",
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

    result_payload = {"games": scoreboard, "players": player_predictions}
    if priced_results:
        result_payload["priced"] = priced_results
    return result_payload


def paper_trade_recent_slates(
    models: Dict[str, Pipeline],
    trainer: Optional["ModelTrainer"],
    config: NFLConfig,
) -> Optional[PaperTradeSummary]:
    if not config.enable_paper_trading:
        return None

    if trainer is None or trainer.feature_builder is None:
        logging.warning("Paper trading skipped: training context unavailable.")
        return None

    feature_builder = trainer.feature_builder
    games = feature_builder.games_frame
    if games is None or games.empty:
        logging.warning("Paper trading skipped: no games ingested for simulation.")
        return None

    if "game_winner" not in models:
        logging.warning("Paper trading skipped: game_winner model not available.")
        return None

    completed = games.copy()
    completed["start_time"] = pd.to_datetime(completed["start_time"], utc=True, errors="coerce")
    completed = completed[
        completed["start_time"].notna()
        & completed["home_score"].notna()
        & completed["away_score"].notna()
    ]
    def _resolve_price_series(frame: pd.DataFrame, side: str) -> pd.Series:
        candidates: List[pd.Series] = []
        for col in (f"{side}_closing_moneyline", f"{side}_moneyline"):
            if col in frame.columns:
                candidates.append(pd.to_numeric(frame[col], errors="coerce"))
        if not candidates:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        series = candidates[0]
        for extra in candidates[1:]:
            series = series.combine_first(extra)
        return series

    completed["_home_price"] = _resolve_price_series(completed, "home")
    completed["_away_price"] = _resolve_price_series(completed, "away")
    price_mask = completed[["_home_price", "_away_price"]].notna().any(axis=1)
    closing_coverage = float(price_mask.mean()) if len(completed) else 0.0
    completed = completed[price_mask]

    lookback_days = max(int(config.paper_trade_lookback_days), 0)
    if lookback_days > 0:
        cutoff = default_now_utc() - dt.timedelta(days=lookback_days)
        completed = completed[completed["start_time"] >= cutoff]

    if completed.empty:
        logging.warning(
            "Paper trading skipped: no completed games with sportsbook odds in the lookback window.")
        return None

    completed = completed.sort_values("start_time").reset_index(drop=True)
    game_features = feature_builder.prepare_upcoming_game_features(completed)
    game_features = game_features.reset_index(drop=True)

    def _ensure_model_features(frame: pd.DataFrame, model: Pipeline) -> pd.DataFrame:
        columns: Optional[Iterable[str]] = getattr(model, "feature_columns", None)
        if not columns:
            target_key = getattr(model, "target_name", None)
            if trainer is not None and target_key:
                columns = trainer.feature_column_map.get(target_key)
        if not columns:
            columns = getattr(model, "feature_names_in_", None)
        if not columns:
            numeric_cols = frame.select_dtypes(include=[np.number, bool]).columns.tolist()
            if not numeric_cols:
                return frame
            return frame.loc[:, numeric_cols]
        frame = frame.copy()
        for col in columns:
            if col not in frame.columns:
                frame[col] = np.nan
        return frame.loc[:, list(columns)]

    winner_model = models["game_winner"]
    winner_features = _ensure_model_features(game_features, winner_model)
    try:
        winner_probs = winner_model.predict_proba(winner_features)[:, 1]
    except Exception:
        logging.exception("Paper trading skipped: unable to score win probabilities for simulation.")
        return None

    if trainer is not None:
        try:
            specials = getattr(trainer, "special_models", {}) or {}
        except Exception:
            specials = {}
        calibration_info = specials.get("game_winner", {}).get("calibration") if isinstance(specials, dict) else None
        if calibration_info and calibration_info.get("method") == "logistic":
            model = calibration_info.get("model")
            if model is not None:
                try:
                    winner_probs = model.predict_proba(winner_probs.reshape(-1, 1))[:, 1]
                except Exception:
                    logging.debug(
                        "Paper trading: unable to apply stored calibration for game_winner", exc_info=True
                    )

    bankroll = float(max(config.paper_trade_bankroll, 0.0))
    max_fraction = float(max(config.paper_trade_max_fraction, 0.0))
    min_edge = float(max(config.paper_trade_edge_threshold, 0.0))

    trades: List[Dict[str, Any]] = []
    total_staked = 0.0
    total_profit = 0.0
    wins = 0
    graded_count = 0

    def _american_prob(value: float) -> float:
        try:
            odds_val = float(value)
        except (TypeError, ValueError):
            return np.nan
        if odds_val >= 0:
            return 100.0 / (odds_val + 100.0)
        return -odds_val / (-odds_val + 100.0)

    for idx, row in completed.iterrows():
        home_odds = row.get("home_closing_moneyline")
        away_odds = row.get("away_closing_moneyline")
        if pd.isna(home_odds):
            home_odds = row.get("home_moneyline")
        if pd.isna(away_odds):
            away_odds = row.get("away_moneyline")
        if pd.isna(home_odds):
            home_odds = row.get("_home_price")
        if pd.isna(away_odds):
            away_odds = row.get("_away_price")
        if pd.isna(home_odds) and pd.isna(away_odds):
            continue

        model_home = float(winner_probs[idx])
        model_home = float(np.clip(model_home, 1e-6, 1 - 1e-6))
        model_away = 1.0 - model_home

        implied_home = _american_prob(home_odds)
        implied_away = _american_prob(away_odds)

        edge_home = model_home - implied_home if not pd.isna(implied_home) else -np.inf
        edge_away = model_away - implied_away if not pd.isna(implied_away) else -np.inf

        options = [
            ("home", model_home, home_odds, implied_home, edge_home),
            ("away", model_away, away_odds, implied_away, edge_away),
        ]
        best_side, model_prob, american_odds, implied_prob, edge_value = max(
            options, key=lambda item: item[4]
        )

        if pd.isna(american_odds) or edge_value < min_edge:
            continue

        stake_fraction = kelly_fraction(model_prob, american_odds, max_fraction)
        if stake_fraction <= 0.0:
            continue

        stake = bankroll * stake_fraction if bankroll > 0 else stake_fraction
        expected_val = ev_of_bet(model_prob, american_odds, stake)
        decimal_price = american_to_decimal(american_odds)

        home_score = float(row.get("home_score", np.nan))
        away_score = float(row.get("away_score", np.nan))
        if pd.isna(home_score) or pd.isna(away_score):
            result = "pending"
            profit = 0.0
        elif home_score == away_score:
            result = "push"
            profit = 0.0
            graded_count += 1
        else:
            home_won = home_score > away_score
            bet_won = home_won if best_side == "home" else not home_won
            profit = stake * (decimal_price - 1.0) if bet_won else -stake
            result = "win" if bet_won else "loss"
            graded_count += 1
            if bet_won:
                wins += 1

        total_staked += stake
        total_profit += profit

        trades.append(
            {
                "game_id": row.get("game_id"),
                "season": row.get("season"),
                "week": row.get("week"),
                "start_time": row.get("start_time"),
                "team_side": best_side,
                "american_odds": american_odds,
                "implied_prob": implied_prob,
                "model_prob": model_prob,
                "edge": edge_value,
                "stake": stake,
                "stake_fraction": stake_fraction,
                "expected_value": expected_val,
                "result": result,
                "profit": profit,
                "closing_bookmaker": row.get("closing_bookmaker"),
            }
        )

    if not trades or total_staked <= 0.0:
        logging.warning("Paper trading produced no actionable bets in the configured window.")
        return None

    roi = total_profit / total_staked if total_staked else 0.0
    hit_rate = (wins / graded_count) if graded_count else float("nan")

    logging.info(
        "Paper trading summary: %d bets | staked %.2f units | profit %.2f | ROI %.2f%% | hit rate %s%%",
        len(trades),
        total_staked,
        total_profit,
        roi * 100.0,
        f"{hit_rate * 100.0:.1f}" if not math.isnan(hit_rate) else "NA",
    )

    trades_df = pd.DataFrame(trades)
    trades_df.sort_values("start_time", inplace=True)

    ledger_path = Path("paper_trades.csv")
    combined_df = trades_df.copy()
    if ledger_path.exists():
        try:
            existing = pd.read_csv(ledger_path)
        except Exception:
            logging.warning("Paper trading: unable to read existing ledger at %s; recreating it.", ledger_path)
        else:
            if not existing.empty:
                if "start_time" in existing.columns:
                    existing["start_time"] = pd.to_datetime(existing["start_time"], errors="coerce")
                combined_df = safe_concat([existing, trades_df], ignore_index=True, sort=False)
                dedupe_keys = ["game_id", "team_side"]
                if "start_time" in combined_df.columns:
                    dedupe_keys.append("start_time")
                dedupe_keys = [col for col in dedupe_keys if col in combined_df.columns]
                if dedupe_keys:
                    combined_df = (
                        combined_df.sort_values("start_time")
                        .drop_duplicates(subset=dedupe_keys, keep="last")
                    )

    combined_df.to_csv(ledger_path, index=False)
    logging.info(
        "Saved paper trading ledger to %s (%d total rows)",
        ledger_path.resolve(),
        len(combined_df),
    )

    settled = combined_df[combined_df["result"].isin(["win", "loss", "push"])]
    cumulative_roi: Optional[float] = None
    if not settled.empty:
        settled = settled.copy()
        settled["stake"] = pd.to_numeric(settled.get("stake"), errors="coerce")
        settled["profit"] = pd.to_numeric(settled.get("profit"), errors="coerce")
        total_staked_all = settled["stake"].fillna(0.0).sum()
        total_profit_all = settled["profit"].fillna(0.0).sum()
        graded = settled[settled["result"].isin(["win", "loss"])]
        graded_count = len(graded)
        if total_staked_all > 0 and graded_count:
            cumulative_roi = total_profit_all / total_staked_all
            logging.info(
                "Paper trading cumulative ROI: %.2f%% across %d graded bets",
                cumulative_roi * 100.0,
                graded_count,
            )

        if "start_time" in settled.columns:
            settled["start_time"] = pd.to_datetime(settled["start_time"], errors="coerce")
            if lookback_days > 0:
                recent_cutoff = default_now_utc() - dt.timedelta(days=lookback_days)
                recent = settled[settled["start_time"] >= recent_cutoff]
            else:
                recent = settled
            recent = recent[recent["result"].isin(["win", "loss"])]
            if not recent.empty:
                recent_staked = recent["stake"].fillna(0.0).sum()
                recent_profit = recent["profit"].fillna(0.0).sum()
                if recent_staked > 0:
                    recent_roi = recent_profit / recent_staked
                    logging.info(
                        "Paper trading %d-day ROI: %.2f%% across %d graded bets",
                        lookback_days,
                        recent_roi * 100.0,
                        len(recent),
                    )

    return PaperTradeSummary(
        ledger=combined_df,
        window_roi=roi,
        cumulative_roi=cumulative_roi,
        graded_bets=graded_count,
        closing_coverage=closing_coverage,
    )


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
    parser.add_argument(
        "--paper-trade",
        action="store_true",
        help="Simulate recent slates with recorded sportsbook odds to validate ROI",
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
    if getattr(args, "paper_trade", False):
        config.enable_paper_trading = True
    setup_logging(config.log_level)

    logging.info("Connecting to PostgreSQL at %s", config.pg_url)
    try:
        engine = create_engine(config.pg_url, future=True)
    except ModuleNotFoundError as exc:
        if exc.name == "psycopg2":
            logging.error(
                "psycopg2 is not installed; install it with `pip install psycopg2-binary` "
                "in your virtualenv before running the script."
            )
            sys.exit(1)
        raise
    db = NFLDatabase(engine)

    try:
        syncer = ClosingOddsArchiveSyncer(config, db)
        syncer.sync()
    except Exception:
        logging.exception("Automated closing odds synchronization failed")

    msf_client = MySportsFeedsClient(
        config.msf_user,
        config.msf_password,
        timeout=config.msf_timeout_seconds,
        timeout_retries=config.msf_timeout_retries,
        timeout_backoff=config.msf_timeout_backoff,
        http_retries=config.msf_http_retries,
    )
    odds_client = OddsApiClient(
        ODDS_API_KEY,
        allow_insecure_ssl=config.odds_allow_insecure_ssl,
    )
    supplemental_loader = SupplementalDataLoader(config)

    # TODO: Backfill automated regression tests that exercise the Odds API ingestion
    # and ROI evaluation pipeline once live prop feeds are verified. The current
    # integration path is only covered by manual end-to-end runs.
    ingestor = NFLIngestor(db, msf_client, odds_client, supplemental_loader)
    ingestor.ingest(config.seasons)

    trainer = ModelTrainer(engine, db, supplemental_loader)
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

    games_frame = getattr(trainer.feature_builder, "games_frame", None)
    if games_frame is None:
        games_frame = pd.DataFrame()

    coverage_frame = games_frame.copy()
    if not coverage_frame.empty:
        time_columns = [
            col
            for col in ("start_time", "game_start", "kickoff", "commence_time")
            if col in coverage_frame.columns
        ]
        kickoff_utc = None
        for col in time_columns:
            parsed = pd.to_datetime(coverage_frame[col], errors="coerce", utc=True)
            if parsed.notna().any():
                kickoff_utc = parsed
                break

        if kickoff_utc is None:
            kickoff_utc = pd.Series(pd.NaT, index=coverage_frame.index)

        coverage_frame = coverage_frame.copy()
        coverage_frame["_kickoff_utc"] = kickoff_utc

        now_utc_coverage = dt.datetime.now(dt.timezone.utc)

        completed_mask = kickoff_utc.notna() & (kickoff_utc <= now_utc_coverage)
        if "status" in coverage_frame.columns:
            statuses = coverage_frame["status"].astype(str).str.lower()
            completed_statuses = {
                "final",
                "finished",
                "complete",
                "completed",
                "closed",
                "finalized",
                "post",
                "postponed",
                "inprogress",
                "in-progress",
                "live",
            }
            upcoming_statuses = {"upcoming", "scheduled", "pre", "pregame"}
            status_mask = statuses.isin(completed_statuses)
            status_mask &= ~statuses.isin(upcoming_statuses)
            completed_mask |= status_mask

        coverage_frame = coverage_frame.loc[completed_mask].copy()
        coverage_frame.drop(columns="_kickoff_utc", inplace=True, errors="ignore")

    def _coverage(frame: pd.DataFrame, columns: Sequence[str]) -> float:
        available_cols = [col for col in columns if col in frame.columns]
        if not available_cols or frame.empty:
            return 0.0
        mask = frame[available_cols].notna()
        return float(mask.all(axis=1).mean())

    def _emit_closing_gap_report(frame: pd.DataFrame) -> Optional[Path]:
        required = {"home_closing_moneyline", "away_closing_moneyline"}
        if frame is None or frame.empty or not required.issubset(frame.columns):
            return None
        missing_mask = frame[list(required)].isna().any(axis=1)
        report_rows = frame.loc[missing_mask]
        report_path = SCRIPT_ROOT / "reports" / "missing_closing_odds.csv"
        if report_rows.empty:
            if report_path.exists():
                try:
                    report_path.unlink()
                except OSError:
                    logging.debug("Unable to remove stale closing odds gap report at %s", report_path)
            return None

        report_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_columns = [
            "season",
            "week",
            "game_id",
            "game_start",
            "start_time",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "home_moneyline",
            "away_moneyline",
            "home_closing_moneyline",
            "away_closing_moneyline",
            "home_closing_implied_prob",
            "away_closing_implied_prob",
            "closing_bookmaker",
            "closing_line_time",
        ]
        available_columns = [col for col in candidate_columns if col in report_rows.columns]
        report = report_rows[available_columns].copy()
        kickoff_utc: Optional[pd.Series] = None

        for candidate in ("game_start", "start_time", "commence_time", "kickoff"):
            if candidate in report_rows.columns:
                parsed = pd.to_datetime(report_rows[candidate], errors="coerce", utc=True)
                if parsed.notna().any():
                    kickoff_utc = parsed
                    if candidate in report.columns:
                        formatted = parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                        report[candidate] = formatted.fillna("")
                    break

        for col in ("game_start", "start_time"):
            if col in report.columns:
                parsed = pd.to_datetime(report[col], errors="coerce", utc=True)
                report[col] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("")

        if kickoff_utc is not None:
            report["kickoff_utc"] = kickoff_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("")
            report["kickoff_date"] = kickoff_utc.dt.strftime("%Y-%m-%d")
            report["kickoff_weekday"] = kickoff_utc.dt.strftime("%a")

        for score_col in ("home_score", "away_score"):
            if score_col in report.columns:
                report[score_col] = pd.to_numeric(report[score_col], errors="coerce")
                report[score_col] = report[score_col].apply(
                    lambda x: int(x) if pd.notna(x) else ""
                )

        sort_priority = [
            col
            for col in (
                "season",
                "week",
                "kickoff_utc",
                "game_start",
                "game_id",
            )
            if col in report.columns
        ]
        if sort_priority:
            report.sort_values(sort_priority, inplace=True)
        try:
            report.to_csv(report_path, index=False)
        except Exception:
            logging.exception("Failed to write closing odds gap report to %s", report_path)
            return None

        logging.warning(
            "Closing odds gap report generated with %d games missing verified closers: %s",
            len(report),
            report_path,
        )
        logging.warning(
            "Populate sportsbook closing prices for these games and rerun ingestion to qualify for live trading."
        )
        return report_path

    def _emit_closing_gap_summary(frame: pd.DataFrame) -> Optional[Path]:
        required = {"home_closing_moneyline", "away_closing_moneyline"}
        if frame is None or frame.empty or not required.issubset(frame.columns):
            return None

        group_columns = [col for col in ("season", "week") if col in frame.columns]
        if not group_columns:
            return None

        working = frame.copy()
        working["_has_closing"] = ~working[list(required)].isna().any(axis=1)

        grouped = working.groupby(group_columns, dropna=False)["_has_closing"].agg(
            total_games="count",
            with_closing="sum",
        )
        summary = grouped.reset_index()
        summary["missing_games"] = summary["total_games"] - summary["with_closing"]
        summary["coverage_pct"] = summary["with_closing"] / summary["total_games"] * 100.0

        if "week" in summary.columns:
            def _format_week(value: Any) -> Any:
                if pd.isna(value):
                    return value
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    return value
                if math.isfinite(numeric) and numeric.is_integer():
                    return int(numeric)
                return value

            summary["week"] = summary["week"].map(_format_week)

        summary.sort_values([col for col in ("coverage_pct", "season", "week") if col in summary.columns], inplace=True)

        summary_path = SCRIPT_ROOT / "reports" / "closing_coverage_summary.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            summary.to_csv(summary_path, index=False)
        except Exception:
            logging.exception("Failed to write closing odds coverage summary to %s", summary_path)
            return None

        missing = summary[summary["missing_games"] > 0]
        if not missing.empty:
            preview = missing.sort_values("coverage_pct").head(5)
            for _, row in preview.iterrows():
                season = row.get("season", "unknown")
                week = row.get("week", "n/a")
                missing_count = int(row["missing_games"])
                total_games = int(row["total_games"])
                coverage_pct = float(row["coverage_pct"])
                logging.warning(
                    "Closing odds coverage gap -> season=%s week=%s: %d of %d games missing closers (coverage %.1f%%).",
                    season,
                    week,
                    missing_count,
                    total_games,
                    coverage_pct,
                )

        return summary_path

    if coverage_frame.empty:
        closing_coverage = 1.0
        rest_coverage = 1.0
        timezone_coverage = 1.0
    else:
        closing_coverage = _coverage(
            coverage_frame, ["home_closing_moneyline", "away_closing_moneyline"]
        )
        rest_coverage = _coverage(
            coverage_frame, ["home_rest_days", "away_rest_days"]
        )
        timezone_coverage = _coverage(
            coverage_frame,
            ["home_timezone_diff_hours", "away_timezone_diff_hours"],
        )

    closing_gap_report: Optional[Path] = None
    closing_gap_summary: Optional[Path] = None
    closing_gap_report_logged = False
    closing_gap_summary_logged = False
    if closing_coverage < 1.0:
        closing_gap_report = _emit_closing_gap_report(coverage_frame)
        closing_gap_summary = _emit_closing_gap_summary(coverage_frame)

    def _log_gap_location() -> None:
        nonlocal closing_gap_report_logged
        if closing_gap_report is not None and not closing_gap_report_logged:
            logging.warning("Missing closing odds are detailed in %s", closing_gap_report)
            closing_gap_report_logged = True

    def _log_summary_location() -> None:
        nonlocal closing_gap_summary_logged
        if closing_gap_summary is not None and not closing_gap_summary_logged:
            logging.warning(
                "Closing odds coverage by season/week written to %s", closing_gap_summary
            )
            closing_gap_summary_logged = True

    if not config.enable_paper_trading and closing_coverage < 0.86:
        logging.warning(
            "Closing odds coverage is %.1f%%. Falling back to paper trading until verified sportsbook closings are loaded.",
            closing_coverage * 100.0,
        )
        config.enable_paper_trading = True
        _log_gap_location()
        _log_summary_location()

    if (
        closing_coverage < 0.86
        or rest_coverage < 0.9
        or timezone_coverage < 0.9
    ):
        if not config.enable_paper_trading:
            logging.warning(
                "Enabling paper trading because data coverage is incomplete (closing=%.1f%%, rest=%.1f%%, timezone=%.1f%%).",
                closing_coverage * 100.0,
                rest_coverage * 100.0,
                timezone_coverage * 100.0,
            )
            config.enable_paper_trading = True
            if closing_coverage < 0.86:
                _log_gap_location()
                _log_summary_location()
        else:
            logging.warning(
                "Data coverage is incomplete (closing=%.1f%%, rest=%.1f%%, timezone=%.1f%%); remain in paper trading mode.",
                closing_coverage * 100.0,
                rest_coverage * 100.0,
                timezone_coverage * 100.0,
            )
            if closing_coverage < 0.86:
                _log_gap_location()
                _log_summary_location()

    paper_summary: Optional[PaperTradeSummary] = None
    if config.enable_paper_trading:
        try:
            paper_summary = paper_trade_recent_slates(models, trainer, config)
            if paper_summary is not None and not paper_summary.ledger.empty:
                sample = paper_summary.ledger.tail(
                    min(5, len(paper_summary.ledger))
                )
                logging.info("Most recent paper trades:\n%s", sample.to_string(index=False))
                logging.info(
                    "Paper trading ledger now contains %d recorded wagers.",
                    len(paper_summary.ledger),
                )
        except Exception:
            logging.exception("Paper trading simulation failed.")

    if paper_summary is None or paper_summary.ledger.empty:
        logging.warning(
            "Paper trading results unavailable; keep the strategy in simulation until closing odds and situational data are complete."
        )
    else:
        graded = paper_summary.graded_bets
        cumulative_roi = paper_summary.cumulative_roi or 0.0
        if (
            paper_summary.closing_coverage < 0.86
            or graded < 50
            or cumulative_roi <= 0.0
        ):
            logging.warning(
                "Live betting disabled: closing coverage=%.1f%%, graded bets=%d, cumulative ROI=%.2f%%. Continue paper trading until odds-aware performance sustains a positive edge.",
                paper_summary.closing_coverage * 100.0,
                graded,
                cumulative_roi * 100.0,
            )
        else:
            logging.info(
                "Paper trading shows %.2f%% cumulative ROI across %d graded bets with %.1f%% odds coverage. Review manually before considering live deployment.",
                cumulative_roi * 100.0,
                graded,
                paper_summary.closing_coverage * 100.0,
            )

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
