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
import json
import logging
import math
import os
import re
import ssl
import time
import unicodedata
import uuid
from collections import defaultdict
from types import SimpleNamespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
from zoneinfo import ZoneInfo

import aiohttp
import numpy as np
import pandas as pd
import requests
from aiohttp import client_exceptions
import certifi
from requests import HTTPError
from requests.auth import HTTPBasicAuth
from requests.exceptions import JSONDecodeError as RequestsJSONDecodeError
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
from urllib.parse import urlencode


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

    def _annotate_keys(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            result = frame.copy()
            result["_event_key"] = pd.Series(dtype=str)
            result["_player_id_key"] = pd.Series(dtype=str)
            result["_player_name_key"] = pd.Series(dtype=str)
            result["_player_team_key"] = pd.Series(dtype=str)
            result["_player_id_event_key"] = pd.Series(dtype=str)
            result["_player_name_event_key"] = pd.Series(dtype=str)
            result["_player_team_event_key"] = pd.Series(dtype=str)
            return result

        keys = frame.apply(_extract_keys, axis=1)
        out = pd.concat([frame.copy(), keys], axis=1)
        mask = out["_player_team_key"].astype(bool)
        out.loc[~mask, "_player_team_key"] = out.loc[~mask, "_player_name_key"]
        mask_event = out["_player_team_event_key"].astype(bool)
        out.loc[~mask_event, "_player_team_event_key"] = out.loc[
            ~mask_event, "_player_name_event_key"
        ]
        return out

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

    if pred_df.empty or odds_df.empty:
        return pd.DataFrame()

    pred_df["_pred_index"] = np.arange(len(pred_df))
    odds_df["_odds_index"] = np.arange(len(odds_df))
    pred_df = pred_df.set_index("_pred_index", drop=False)
    odds_df = odds_df.set_index("_odds_index", drop=False)

    allowed_side_map = {
        "anytime_td": {"yes", "over"},
        "passing_yards": {"over"},
        "receiving_yards": {"over"},
        "receptions": {"over"},
        "rushing_yards": {"over"},
    }

    def _merge_on_key(
        preds: pd.DataFrame, offers: pd.DataFrame, key_col: str
    ) -> pd.DataFrame:
        if preds.empty or offers.empty:
            return pd.DataFrame()

        key_cols = ["market", key_col]
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
            best["_side_norm"] = best["side"].fillna("").str.lower()
            best = best[
                best.apply(
                    lambda row: row["_side_norm"]
                    in allowed_side_map.get(row["market"], {row["_side_norm"]}),
                    axis=1,
                )
            ].drop(columns=["_side_norm"], errors="ignore")
            if best.empty:
                return pd.DataFrame()

        join_cols = [
            col for col in key_cols if col in preds.columns and col in best.columns
        ]
        if not join_cols:
            join_cols = ["market", key_col]

        return preds.merge(
            best,
            on=join_cols,
            how="inner",
            suffixes=("", "_book"),
        )

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
        merged_slice = _merge_on_key(preds_slice, offers_slice, key_col)
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

NFL_API_USER = "4359aa1b-cc29-4647-a3e5-7314e2"
NFL_API_PASS = "MYSPORTSFEEDS"

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
    respect_lineups: bool = True
    odds_allow_insecure_ssl: bool = env_flag("ODDS_ALLOW_INSECURE_SSL", False)

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
            Column("odds_event_id", String),
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
                }
            )
        return results

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
        if advanced_rows_map:
            self.db.upsert_rows(
                self.db.team_advanced_metrics,
                list(advanced_rows_map.values()),
                ["metric_id"],
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
        games_by_key: Dict[Tuple[str, str, Optional[dt.date]], List[Dict[str, Any]]] = defaultdict(list)
        min_start: Optional[dt.datetime] = None
        max_start: Optional[dt.datetime] = None

        for row in game_lookup_rows:
            start_time = _ensure_datetime(row.get("start_time"))
            if start_time is not None:
                if min_start is None or start_time < min_start:
                    min_start = start_time
                if max_start is None or start_time > max_start:
                    max_start = start_time
            home = normalize_team_abbr(row.get("home_team"))
            away = normalize_team_abbr(row.get("away_team"))
            if home and away:
                key = (home, away, start_time.date() if start_time else None)
                games_by_key[key].append(row)
            event_id = row.get("odds_event_id")
            if event_id:
                games_by_event[str(event_id)] = row

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

        for event in odds_data:
            event_id = str(event.get("id") or "")
            if not event_id:
                continue

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
            last_update = parse_dt(primary_book.get("last_update"))
            moneyline_market = None
            for market in primary_book.get("markets", []) or []:
                if (market.get("key") or "").lower() == "h2h":
                    moneyline_market = market
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

            odds_rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "start_time": commence_time or _ensure_datetime(matched_game.get("start_time")),
                    "home_team": home_team,
                    "away_team": away_team,
                    "status": matched_game.get("status") or "scheduled",
                    "home_moneyline": home_price,
                    "away_moneyline": away_price,
                    "home_implied_prob": _american_to_prob(home_price),
                    "away_implied_prob": _american_to_prob(away_price),
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
        self.latest_odds_lookup: Optional[pd.DataFrame] = None
        self.game_totals_frame: Optional[pd.DataFrame] = None
        self.player_prop_lines_frame: Optional[pd.DataFrame] = None

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

    def _backfill_game_odds(self, games: pd.DataFrame) -> pd.DataFrame:
        if games is None or games.empty:
            return games

        working = games.copy()
        if "start_time" in working.columns:
            working["start_time"] = pd.to_datetime(working["start_time"], errors="coerce")
            working["_start_date"] = working["start_time"].dt.normalize()

        odds_cols = [
            "home_moneyline",
            "away_moneyline",
            "home_implied_prob",
            "away_implied_prob",
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

        working.drop(columns=["_start_date"], inplace=True, errors="ignore")
        return working

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

                def _group_stats(frame: pd.DataFrame, cols: List[str]) -> Dict[Any, Dict[str, float]]:
                    if frame.empty:
                        return {}
                    stats: Dict[Any, Dict[str, float]] = {}
                    for key, group in frame.groupby(cols):
                        weights = group["_usage_weight"].clip(lower=1e-4)
                        values = group[target].astype(float)
                        if weights.sum() <= 0:
                            weights = pd.Series(1.0, index=values.index)
                        mean_val = float(np.average(values, weights=weights))
                        stats[key if isinstance(key, tuple) else key] = {
                            "mean": mean_val,
                            "weight": float(weights.sum()),
                        }
                    return stats

                team_pos_prior = _group_stats(labeled, ["team", "position"])
                pos_prior = _group_stats(labeled, ["position"])
                league_weights = labeled["_usage_weight"].clip(lower=1e-4)
                league_mean = float(np.average(labeled[target].astype(float), weights=league_weights))
                league_weight = float(league_weights.sum())

                placeholders = subset_all[subset_all[target].isna()].copy()
                synthetic_rows: List[Dict[str, Any]] = []
                if not placeholders.empty:
                    for _, row in placeholders.iterrows():
                        team_key = (row.get("team"), row.get("position"))
                        pos_key = row.get("position")
                        numerator = 0.0
                        weight_sum = 0.0
                        effective_weight = 0.0

                        team_stats = team_pos_prior.get(team_key)
                        if team_stats:
                            w = max(team_stats["weight"], 1e-4)
                            numerator += team_stats["mean"] * w
                            weight_sum += w
                            effective_weight += w

                        pos_stats = pos_prior.get(pos_key)
                        if pos_stats:
                            w = max(pos_stats["weight"] * 0.5, 1e-4)
                            numerator += pos_stats["mean"] * w
                            weight_sum += w
                            effective_weight += pos_stats["weight"]

                        if league_weight > 0:
                            league_w = max(league_weight * 0.25, 1e-4)
                            numerator += league_mean * league_w
                            weight_sum += league_w
                            effective_weight += league_weight

                        if weight_sum <= 0:
                            continue

                        target_estimate = numerator / weight_sum
                        synthetic = row.to_dict()
                        synthetic[target] = float(target_estimate)
                        synthetic["is_synthetic"] = True
                        influence = effective_weight / (effective_weight + 25.0)
                        synthetic["sample_weight"] = float(np.clip(influence, 0.05, 0.4))
                        synthetic_rows.append(synthetic)

                if synthetic_rows:
                    logging.debug(
                        "Generated %d prior rows for %s but leaving them out of model training",
                        len(synthetic_rows),
                        target,
                    )

                combined = labeled.drop(columns=["_usage_weight"], errors="ignore")
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
            if col in games_context.columns:
                games_context[col] = games_context[col].fillna(0.0)

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
                        row_copy["travel_penalty"] = strength.get("travel_penalty")
                        row_copy["rest_penalty"] = strength.get("rest_penalty")
                        row_copy["weather_adjustment"] = strength.get("weather_adjustment")
                        row_copy["avg_timezone_diff_hours"] = strength.get("avg_timezone_diff_hours")

                    history = self._get_latest_team_history(team, season)
                    if history is not None:
                        if pd.isna(row_copy.get("rest_penalty")):
                            row_copy["rest_penalty"] = history.get("rest_penalty")
                        if pd.isna(row_copy.get("travel_penalty")):
                            row_copy["travel_penalty"] = history.get("travel_penalty")
                        if pd.isna(row_copy.get("avg_timezone_diff_hours")):
                            row_copy["avg_timezone_diff_hours"] = history.get("timezone_diff_hours")

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

                    opp_history = self._get_latest_team_history(opponent, season)
                    if opp_history is not None:
                        if pd.isna(row_copy.get("opp_rest_penalty")):
                            row_copy["opp_rest_penalty"] = opp_history.get("rest_penalty")
                        if pd.isna(row_copy.get("opp_travel_penalty")):
                            row_copy["opp_travel_penalty"] = opp_history.get("travel_penalty")
                        if pd.isna(row_copy.get("opp_timezone_diff_hours")):
                            row_copy["opp_timezone_diff_hours"] = opp_history.get("timezone_diff_hours")

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
            if col in player_features.columns:
                player_features[col] = player_features[col].fillna(0.0)

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
        self.target_priors: Dict[str, Dict[str, Any]] = {}
        self.prior_engines: Dict[str, Optional[Dict[str, Any]]] = {}
        self.special_models: Dict[str, Dict[str, Any]] = {}
        self.feature_column_map: Dict[str, List[str]] = {}

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
            "league": {"mean": np.nan, "weight": 0.0},
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
        priors["league"] = {
            "mean": float(np.average(values_all, weights=weights_all)),
            "weight": float(weights_all.sum()),
        }

        for (team, position), group in actual.groupby(["team", "position"]):
            group_weights = (
                group.get("sample_weight", pd.Series(1.0, index=group.index))
                .astype(float)
                .clip(lower=1e-4)
            )
            group_values = group[target].astype(float)
            priors["team_position"][(team, position)] = {
                "mean": float(np.average(group_values, weights=group_weights)),
                "weight": float(group_weights.sum()),
            }

        for position, group in actual.groupby(["position"]):
            group_weights = (
                group.get("sample_weight", pd.Series(1.0, index=group.index))
                .astype(float)
                .clip(lower=1e-4)
            )
            group_values = group[target].astype(float)
            priors["position"][position] = {
                "mean": float(np.average(group_values, weights=group_weights)),
                "weight": float(group_weights.sum()),
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

                if moneyline_col not in working.columns:
                    working[moneyline_col] = np.nan
                working[moneyline_col] = pd.to_numeric(
                    working.get(moneyline_col), errors="coerce"
                )

                if implied_col not in working.columns:
                    working[implied_col] = np.nan
                working[implied_col] = pd.to_numeric(
                    working.get(implied_col), errors="coerce"
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
        closing_cols = [c for c in out_df.columns if c.startswith("line_") or c.endswith("_implied_prob")]
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
            # join to implied probs derived from moneylines already in games table
            # Expect columns like 'home_implied_prob'/'away_implied_prob' from ingest (see _ingest_odds)
            side = out_df.get("team_side")  # optional column you may carry ("home"/"away")
            if side is not None and "home_implied_prob" in out_df and "away_implied_prob" in out_df:
                cl = np.where(side == "home", out_df["home_implied_prob"], out_df["away_implied_prob"])
                mae_vs_closing = float(np.nanmean(np.abs(out_df["_y_pred"] - cl)))
        elif closing_cols:
            # e.g., player props with 'line_receiving_yards', etc.
            # Choose the first matching line as a reference
            ref = out_df[closing_cols[0]].astype(float)
            mae_vs_closing = float(np.nanmean(np.abs(out_df["_y_pred"] - ref)))

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

            required_cols = {"home_moneyline", "away_moneyline", "home_score", "away_score"}
            missing_required = required_cols - set(out_df.columns)
            if missing_required:
                logging.debug(
                    "%s: skipping edge evaluation (%s missing)",
                    target,
                    ", ".join(sorted(missing_required)),
                )
                return None

            price_cols = ["home_moneyline", "away_moneyline"]
            market_frame = out_df.loc[:, price_cols].apply(pd.to_numeric, errors="coerce")
            pred_series = pd.Series(preds_array, index=out_df.index, dtype=float)

            home_prices = market_frame["home_moneyline"].astype(float)
            away_prices = market_frame["away_moneyline"].astype(float)

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

            synthetic_odds_used = False
            # TODO: Ensure historical odds are persisted prior to evaluation so that
            # live ROI metrics no longer rely on this synthetic -110 moneyline
            # fallback. Once genuine prices are available, this branch should become
            # an explicit opt-in emergency path.
            if not valid_mask.any():
                logging.debug(
                    "%s: no rows with market odds; using synthetic -110 moneylines for diagnostics",
                    target,
                )
                synthetic_odds_used = True
                fallback_prices = pd.Series(-110.0, index=out_df.index, dtype=float)
                implied = _american_to_prob(fallback_prices)
                home_prices = fallback_prices.copy()
                away_prices = fallback_prices.copy()
                home_probs = implied.copy()
                away_probs = implied.copy()
                valid_mask = results_mask.copy()
                home_valid = valid_mask.copy()
                away_valid = valid_mask.copy()

            if not valid_mask.any():
                logging.debug("%s: no rows with complete market data for edge check", target)
                return None

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
                    picks.append(
                        home_df[["p_model", "side_win", "payout", "_edge", "_threshold", "_side", "_is_push"]]
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
                    picks.append(
                        away_df[["p_model", "side_win", "payout", "_edge", "_threshold", "_side", "_is_push"]]
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
            if synthetic_odds_used:
                label_suffix = f"{label_suffix}|synthetic_odds"

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
            return {
                "roi": roi_val,
                "roi_lo": roi_lo_val,
                "roi_hi": roi_hi_val,
                "hit_rate": hit_val,
                "n_bets": bet_count,
                "label": label_suffix,
            }

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
                label_suffix = f"{label_suffix}|priors_baseline|synthetic_odds"

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
            }
        self.db.record_backtest_metrics(self.run_id, target, summary, sample_size=int(len(out_df)))

        logging.info(
            "%s %s | folds=%d | MAE=%.3f | RMSE=%.3f | MAE_vs_close=%s | ROI=%s (CI=%s) | bets=%d | edge_source=%s",
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
