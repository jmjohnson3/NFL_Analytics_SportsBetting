import pytest

pd = pytest.importorskip("pandas")
sqlalchemy = pytest.importorskip("sqlalchemy")
create_engine = sqlalchemy.create_engine

from NFL_SPORTSBETTING import FeatureBuilder


def _build_builder(player_games: pd.DataFrame) -> FeatureBuilder:
    engine = create_engine("sqlite://")
    builder = FeatureBuilder(engine)
    builder.player_feature_frame = player_games
    return builder


def _upcoming_game(home_team: str, away_team: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "test-game",
                "start_time": pd.Timestamp("2024-09-01T17:00:00Z"),
                "home_team": home_team,
                "away_team": away_team,
                "season": "2024",
                "week": 1,
                "venue": "Test Field",
                "city": "Test City",
                "state": "TS",
                "day_of_week": "Sunday",
            }
        ]
    )


def test_defensive_splits_adjust_projections(monkeypatch):
    monkeypatch.setenv("NFL_ENABLE_DEFENSIVE_SPLITS", "1")

    player_games = pd.DataFrame(
        [
            {
                "player_id": 1,
                "player_name": "Test QB",
                "team": "NYJ",
                "position": "QB",
                "game_id": "hist-1",
                "start_time": pd.Timestamp("2023-10-01T17:00:00Z"),
                "passing_yards": 300,
                "passing_attempts": 35,
            },
            {
                "player_id": 1,
                "player_name": "Test QB",
                "team": "NYJ",
                "position": "QB",
                "game_id": "hist-2",
                "start_time": pd.Timestamp("2023-10-08T17:00:00Z"),
                "passing_yards": 320,
                "passing_attempts": 36,
            },
        ]
    )

    builder = _build_builder(player_games)

    upcoming_games = _upcoming_game("NYJ", "NYG")

    softer_defense = pd.DataFrame(
        [
            {
                "defense_team": "NYG",
                "position_group": "QB",
                "passing_yards_multiplier": 0.8,
            }
        ]
    )

    stiffer_defense = pd.DataFrame(
        [
            {
                "defense_team": "NYG",
                "position_group": "QB",
                "passing_yards_multiplier": 1.2,
            }
        ]
    )

    builder.defensive_splits_frame = softer_defense
    softer_projection = builder.prepare_upcoming_player_features(
        upcoming_games, starters_per_position={"QB": 1}
    )
    assert not softer_projection.empty
    softer_value = softer_projection.loc[
        softer_projection["player_id"] == 1, "passing_yards"
    ].iloc[0]

    builder.defensive_splits_frame = stiffer_defense
    stiffer_projection = builder.prepare_upcoming_player_features(
        upcoming_games, starters_per_position={"QB": 1}
    )
    assert not stiffer_projection.empty
    stiffer_value = stiffer_projection.loc[
        stiffer_projection["player_id"] == 1, "passing_yards"
    ].iloc[0]

    assert stiffer_value != softer_value
    assert stiffer_value > softer_value
