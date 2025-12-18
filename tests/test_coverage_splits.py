import pytest

try:
    from NFL_SPORTSBETTING import FeatureBuilder
    from NFL_SPORTSBETTING import NFLConfig, SupplementalDataLoader
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependency guard
    pytest.skip(f"NFL_SPORTSBETTING dependency missing: {exc}", allow_module_level=True)

sqlalchemy = pytest.importorskip("sqlalchemy")
create_engine = sqlalchemy.create_engine
pd = pytest.importorskip("pandas")


def _builder_with_history(player_games: pd.DataFrame) -> FeatureBuilder:
    engine = create_engine("sqlite://")
    builder = FeatureBuilder(engine)
    builder.player_feature_frame = player_games
    return builder


def _upcoming(home_team: str, away_team: str) -> pd.DataFrame:
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


def test_coverage_splits_adjust_projections(monkeypatch):
    monkeypatch.setenv("NFL_ENABLE_COVERAGE_SPLITS", "1")

    player_games = pd.DataFrame(
        [
            {
                "player_id": 10,
                "player_name": "Test WR",
                "team": "NYJ",
                "position": "WR",
                "game_id": "hist-1",
                "start_time": pd.Timestamp("2023-10-01T17:00:00Z"),
                "receiving_targets": 8,
                "receiving_yards": 90,
                "receptions": 6,
            },
            {
                "player_id": 10,
                "player_name": "Test WR",
                "team": "NYJ",
                "position": "WR",
                "game_id": "hist-2",
                "start_time": pd.Timestamp("2023-10-08T17:00:00Z"),
                "receiving_targets": 9,
                "receiving_yards": 110,
                "receptions": 7,
            },
        ]
    )

    coverage_splits = pd.DataFrame(
        [
            {
                "defense_team": "NYG",
                "coverage_type": "man",
                "coverage_pct": 0.7,
                "position_group": "WR",
                "receiving_targets_multiplier": 1.1,
                "receiving_yards_multiplier": 1.2,
                "receptions_multiplier": 1.05,
            },
            {
                "defense_team": "NYG",
                "coverage_type": "zone",
                "coverage_pct": 0.3,
                "position_group": "WR",
                "receiving_targets_multiplier": 1.0,
                "receiving_yards_multiplier": 1.0,
                "receptions_multiplier": 1.0,
            },
            {
                "defense_team": "PHI",
                "coverage_type": "man",
                "coverage_pct": 0.2,
                "position_group": "WR",
                "receiving_targets_multiplier": 0.9,
                "receiving_yards_multiplier": 0.85,
                "receptions_multiplier": 0.95,
            },
            {
                "defense_team": "PHI",
                "coverage_type": "zone",
                "coverage_pct": 0.8,
                "position_group": "WR",
                "receiving_targets_multiplier": 0.85,
                "receiving_yards_multiplier": 0.8,
                "receptions_multiplier": 0.9,
            },
        ]
    )

    builder = _builder_with_history(player_games)
    builder.coverage_splits_frame = coverage_splits

    vs_man_heavy = builder.prepare_upcoming_player_features(
        _upcoming("NYJ", "NYG"), starters_per_position={"WR": 3}
    )
    vs_zone_heavy = builder.prepare_upcoming_player_features(
        _upcoming("NYJ", "PHI"), starters_per_position={"WR": 3}
    )

    assert not vs_man_heavy.empty
    assert not vs_zone_heavy.empty

    man_yards = vs_man_heavy.loc[vs_man_heavy["player_id"] == 10, "receiving_yards"].iloc[0]
    zone_yards = vs_zone_heavy.loc[vs_zone_heavy["player_id"] == 10, "receiving_yards"].iloc[0]

    assert man_yards != zone_yards
    assert man_yards > zone_yards


def test_coverage_splits_api_loader(monkeypatch):
    requests = pytest.importorskip("requests")

    sample_csv = """team,man_yards_per_target,man_coverage_pct,zone_yards_per_target,zone_coverage_pct,man_targets_allowed,zone_targets_allowed,man_receptions_allowed,zone_receptions_allowed
NYG,9.5,0.6,7.5,0.4,40,30,22,18
PHI,6.8,0.3,7.2,0.7,30,50,18,30
"""

    class DummyResponse:
        status_code = 200
        text = sample_csv

        def raise_for_status(self):
            return None

    def fake_get(url, timeout=None):
        assert "coverage" in url
        return DummyResponse()

    monkeypatch.setenv("NFL_ENABLE_COVERAGE_SPLITS", "1")
    monkeypatch.setenv(
        "NFL_COVERAGE_SPLITS_API_URL", "https://example.com/coverage.csv"
    )
    monkeypatch.setattr(requests, "get", fake_get, raising=False)

    config = NFLConfig()
    loader = SupplementalDataLoader(config)
    frame = loader.coverage_splits_frame

    assert not frame.empty
    nyg_rows = frame[frame["defense_team"] == "NYG"]
    assert not nyg_rows.empty
    assert nyg_rows["receiving_yards_multiplier"].notna().any()
