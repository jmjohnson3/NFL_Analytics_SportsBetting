# NFL Analytics Sports Betting

## Why to Hold Off for Now

The player feature builder no longer fabricates synthetic "prior" rows when stat lines
are missing. Instead, the training datasets only include verified outcomes and emit a
warning when gaps remain. Until those historical rows are populated, the models will
have blind spots that paper trading should surface immediately.

Situational features also require real inputs. Travel, rest, weather, and timezone
adjustments default to `NaN` until you supply the supplemental CSV. Models trained on
missing context will not reflect real-world edges, and the runtime coverage checks keep
the system in simulation mode until those tables are filled.

Finally, backtests now demand authentic sportsbook closing lines. If a completed game
does not have a recorded bookmaker close, the script logs the gap, treats the matchup
as uncovered, and refuses to enter live mode. Without that data you cannot audit
`MAE_vs_close` or ROI, so the guardrails treat the system as paper-trade only.

## Prerequisites Before Betting Real Money

1. **Backfill real sportsbook odds** for all historical games that fall in your
   training and validation windows so each completed matchup has recorded moneylines.
2. **Populate rest, travel, timezone, and weather inputs** (or drop the features) so
   the feature warnings disappear because the data exist, not because they are masked.
3. **Paper-trade future slates** once the data gaps above are closed and confirm that
   the live ROI remains positive against the true lines you intend to bet.

Only after these gaps are addressed should you consider deploying the model in a live
betting environment.

### Interpreting the new coverage warnings

When you run `python NFL_SPORTSBETTING.py`, the driver inspects the assembled games
frame and measures how many rows have authentic closing moneylines, rest metrics, and
timezone adjustments. If you try to disable paper trading while closing coverage is
below 90%, the run aborts with an explicit error. Otherwise, the script automatically
forces paper-trade mode, prints the exact percentages, and refuses to surface a
live-deployment summary. A log entry such as:

```
WARNING | root | Enabling paper trading because data coverage is incomplete (closing=3.1%, rest=0.0%, timezone=0.0%).
WARNING | root | Paper trading results unavailable; keep the strategy in simulation until closing odds and situational data are complete.
```

means the supplemental CSVs are still mostly empty. Until those warnings disappear and
the historical paper-trading ledger shows a sustained, odds-aware edge, the project is
not ready for real-money wagering. Each run now also emits `reports/missing_closing_odds.csv`
whenever the coverage check finds unmatched games. The file lists the season, week,
teams, and whatever odds data already exist for quick reconciliation. The report now
adds explicit kickoff metadata&mdash;`kickoff_utc`, `kickoff_date`, and
`kickoff_weekday`&mdash;plus the recorded final scores (when available) so you can
line up each matchup against your external odds source without diving back into the
database. Fill in the bookmaker closing lines there (or import them into your database)
and rerun the pipeline until the closing coverage passes 90%. When large portions of
history are uncovered, the driver also writes `reports/closing_coverage_summary.csv`,
which aggregates coverage by season/week so you can target the exact windows that still
need verified closes or adjust the configuration to a span where authentic pricing is
available.

## New data hooks

The application now exposes two CSV-driven overrides to make the gaps above explicit
and auditable:

- `data/closing_odds_history.csv` &mdash; populate this with bookmaker closing prices
  (moneylines and implied probabilities) for every historical matchup you train on.
  The ingestion pipeline will merge these values into the `nfl_games` table and the
  evaluation routines will refuse to leave paper-trading mode until closing coverage
  exceeds 90%.
- `data/team_travel_context.csv` &mdash; record verified rest days, travel penalties,
  and timezone adjustments for each team/week. These values flow directly into the
  situational feature set so the models stop learning on `NaN` placeholders.

Both files ship with example rows to illustrate the required schema. Replace the
samples with real, validated records before relying on any model output.

### If you cannot locate closing odds for certain games

The guardrails are intentionally strict: any matchup without a verified
bookmaker close keeps the entire run in paper-trade mode. When the coverage
report lists games you cannot immediately source, use one of the following
approaches:

1. **Track down an alternative feed.** Many historical odds vendors (e.g.,
   SportsOddsHistory, KillerSports, licensed sportsbook data products) archive
   closing prices. Import those values into `data/closing_odds_history.csv` (or
   your database) so the evaluation metrics stay grounded in real markets.
2. **Manually enter vetted closes.** If you have access to trusted screenshots
   or settlement reports, convert those into the CSV format and mark the
   `bookmaker` and `close_timestamp` columns accordingly. Leave rows blank rather
   than guessing; a missing value is safer than an invented one.
3. **Narrow the training window.** When certain seasons lack dependable pricing,
   adjust your configuration so backtests only span the period where authentic
   closes exist. The coverage check will pass once at least 90% of the games in
   scope contain real closing odds.
4. **Stay in simulation.** If genuine closing numbers cannot be obtained for the
   relevant window, do not disable the paper-trade requirement. The models would
   be benchmarking against assumptions instead of the odds you actually face.

### What if the coverage never reaches 90%?

When closing odds remain missing after you exhaust the steps above, the runtime
guardrails simply keep you in paper-trade mode:

- The driver continues to emit warnings that list the exact seasons and weeks
  where closes are absent.
- `reports/missing_closing_odds.csv` and `reports/closing_coverage_summary.csv`
  are regenerated on every run so you can revisit the gaps later.
- No live-deployment summary is produced, and any attempt to override paper
  trading is blocked while coverage stays below the 90% threshold.

In other words, the application assumes the data are incomplete and refuses to
endorse live staking. Continue operating in simulation until you can supply
authentic bookmaker closes or shrink the evaluation window to a span where the
market data are trustworthy.

The key principle is that you should never fabricate or forward-fill closing
prices. Either find the real numbers or exclude the affected games from any
evaluation you intend to trust for live betting decisions.

### Automating the closing-odds backfill

To simplify the backfill, the driver script can now download and normalize
archived moneylines directly from supported vendors before every run. Configure
the provider via environment variables or a `.env` file before launching the
pipeline:

| Setting | Purpose |
| --- | --- |
| `NFL_CLOSING_ODDS_PROVIDER` | Optional. Defaults to `sportsoddshistory`; set to `killersports` to use that feed or `none`/empty to disable the automatic sync. |
| `NFL_CLOSING_ODDS_TIMEOUT` | Optional HTTP timeout (seconds). Defaults to `45`. |
| `NFL_CLOSING_ODDS_DOWNLOAD_DIR` | Optional folder where raw payloads are cached for auditing. |
| `SPORTSODDSHISTORY_BASE_URL` | Override the default SportsOddsHistory endpoint if needed. |
| `SPORTSODDSHISTORY_USER_AGENT` | Custom User-Agent string when hitting SportsOddsHistory. |
| `KILLERSPORTS_BASE_URL` | Base URL for KillerSports exports (required when that provider is selected). |
| `KILLERSPORTS_API_KEY` | Bearer token for KillerSports, when their API requires it. |
| `KILLERSPORTS_USERNAME` / `KILLERSPORTS_PASSWORD` | HTTP basic credentials for KillerSports, if applicable. |

Example shell snippet for SportsOddsHistory (now the default):

```bash
export NFL_CLOSING_ODDS_TIMEOUT=60
python NFL_SPORTSBETTING.py
```

To skip the automatic download entirely and rely on a manually curated
`data/closing_odds_history.csv`, explicitly clear the provider before running:

```bash
export NFL_CLOSING_ODDS_PROVIDER=none
python NFL_SPORTSBETTING.py
```

Example for KillerSports (when hosted behind basic auth):

```bash
export NFL_CLOSING_ODDS_PROVIDER=killersports
export KILLERSPORTS_BASE_URL="https://api.killersports.com/nfl/closing"
export KILLERSPORTS_USERNAME="my-user"
export KILLERSPORTS_PASSWORD="my-secret"
python NFL_SPORTSBETTING.py
```

During startup the script downloads the requested seasons, normalizes the
payload into the schema used by `data/closing_odds_history.csv`, and merges the
result with the local history file (deduplicating by season, week, and teams).
If the provider fails, the code logs a warning and falls back to whatever odds
already exist in the CSV. You can therefore seed the CSV manually, automate the
pull via these providers, or combine both approaches. The coverage reports and
paper-trade guardrails continue to operate exactly as before; the automated
sync merely helps you reach the 90% threshold with less manual data entry.
