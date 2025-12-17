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

### Odds API configuration

The optional live odds fetcher in `odds_extract.py` (and the main driver)
ships with the key `5b6f0290e265c3329b3ed27897d79eaf` already hard-coded so
ingestion works out of the box. Replace this value with your own The Odds API
credential if needed; leaving it untouched keeps the bundled key in use for
both live and historical odds pulls.

Both files ship with example rows to illustrate the required schema. Replace the
samples with real, validated records before relying on any model output.

### Player projection smoothing

Player stat forecasts now explicitly down-weight single-game volatility. For each
player we blend three signals: the most recent game (~5%), a rolling five-game
average (~25% scaled down when fewer games are available), and the full-season
average for stability (~70%). Forecasts are also capped to roughly a 95th
percentile of the player's own historical production (with modest headroom) so
backups cannot inherit starter-level stat lines. You can adjust the weights or
window in
`NFL_SPORTSBETTING.py` via `RECENT_FORM_LAST_GAME_WEIGHT`,
`RECENT_FORM_WINDOW_WEIGHT`, and `RECENT_FORM_GAMES` if you want a different
balance between short- and long-term form. These values (and the historical cap
quantile/headroom) can also be set without editing code by supplying a JSON
config or environment variables:

- `NFL_RECENT_FORM_GAMES`
- `NFL_RECENT_FORM_LAST_WEIGHT`
- `NFL_RECENT_FORM_WINDOW_WEIGHT`
- `NFL_PLAYER_HISTORY_CAP_QUANTILE`
- `NFL_PLAYER_HISTORY_CAP_HEADROOM`

If the supplied weights exceed 100% of the blend, the driver now rescales them
and logs the corrected values so the season-average anchor remains positive.

Player prop outputs now mirror the game-level tables by carrying `confidence`
labels, a `consensus_gap` (model minus market implied probability), and an
`action` recommendation (`target`, `lean`, `monitor`, or `pass`). These columns
are included in the priced CSVs and the "Top player props" log snippet.

### If you cannot locate closing odds for certain games

The guardrails are intentionally strict: any matchup without a verified
bookmaker close keeps the entire run in paper-trade mode. When the coverage
report lists games you cannot immediately source, use one of the following
approaches:

1. **Track down an alternative feed.** Many historical odds vendors (e.g.,
   OddsPortal exports, KillerSports, licensed sportsbook data products) archive
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

### Choosing a closing-odds source

You now have two supported options for filling closing moneylines:

1) **Local CSV (default).** Point `NFL_CLOSING_ODDS_PATH` at your
   `odds_history.csv` (the bundled `data/closing_odds_history.csv` sample still
   works). Set `NFL_CLOSING_ODDS_PROVIDER=local` (or leave unset) to rely solely
   on your archive.

2) **The Odds API.** Set `NFL_CLOSING_ODDS_PROVIDER=oddsapi` and provide the
   following environment variables to download closes directly from
   https://the-odds-api.com instead of scraping:

   - `NFL_ODDS_API_KEY` – your The Odds API key (required)
   - `NFL_ODDS_API_SPORT` – defaults to `americanfootball_nfl`
   - `NFL_ODDS_API_REGIONS` – defaults to `us`
   - `NFL_ODDS_API_MARKETS` – defaults to `h2h` (moneyline)
   - `NFL_ODDS_API_BOOKMAKER` – optional bookmaker key to prefer
   - `NFL_ODDS_API_FORMAT` – defaults to `american`
   - `NFL_ODDS_API_SNAPSHOT` – optional ISO timestamp to pull historical
     snapshots (paid plans only)

Historical snapshots via The Odds API require a paid plan; without the snapshot
parameter the provider fetches the latest available odds for each event.

## Play-by-play simulation vs. current scope

This repo does **not** attempt to forecast every snap in sequence. The models are
trained and calibrated at the game and player-prop level, which is the granularity
of the available historical feeds. Predicting each play would require a different
data asset (full play-by-play with personnel, formation, coverage, and motion
labels), a stateful simulator that updates win probability after every snap, and
an action-policy model to decide run/pass tendencies on the fly. If you need more
granular script sensitivity today, the closest option is to:

- Run the existing game-level projections to get expected score margin and total.
- Use those margins to scale team rush/pass rates (already supported in
  `NFL_SPORTSBETTING.py`) so trailing teams throw more and leading teams run more.
- Optionally layer in opponent positional strengths (e.g., CB/edge/IDL ratings) to
  tilt targets and yards per play for specific skill players.

With the current inputs, this yields drive-level realism—who scores, by how much,
and how usage shifts when a team is leading or trailing—without pretending to know
the exact sequence of snaps. To move toward true play-by-play simulation, you would
first need reliable historical play-level data, then fit a conditional play-caller
model that can roll forward the game state on every down.

### Automating the closing-odds backfill

The pipeline now performs a simple CSV ingest before each run—no HTTP calls or
scraping are performed. Configure the location of your odds archive via
environment variables or a `.env` file:

| Setting | Purpose |
| --- | --- |
| `NFL_CLOSING_ODDS_PROVIDER` | Defaults to `local`; other values are ignored now that scraping has been removed. |
| `NFL_CLOSING_ODDS_PATH` | Optional path to your `odds_history.csv` file. Defaults to `data/closing_odds_history.csv` when present. |

If the CSV is missing or empty, closing odds will stay blank until you supply a
valid file.

### Optional coverage-specific player tweaks

If you track how players perform against different defensive coverages, you can
layer those tendencies onto the model outputs without retraining. The loader now
prefers an API or a scraped HTML table so you do not need to maintain CSVs:

- Point `NFL_COVERAGE_API_BASE` to an API host and (optionally) set
  `NFL_COVERAGE_API_KEY`. The driver will request
  `<base>/<NFL_COVERAGE_API_PLAYER_ENDPOINT>` (defaults to
  `player-adjustments`) for player rules and
  `<base>/<NFL_COVERAGE_API_TEAM_ENDPOINT>` (defaults to `team-coverage`) for
  team schemes. Responses should include columns/fields `player` or `team`,
  `coverage_type` (`man`/`zone`), and `adjustment_pct`.
- Alternatively, set `NFL_COVERAGE_SCRAPE_PLAYER_URL` and
  `NFL_COVERAGE_SCRAPE_TEAM_URL` to pages containing HTML tables with those same
  columns; the code will scrape and normalize them automatically.
- CSVs remain optional fallbacks. If you still keep files around, you can point
  `NFL_COVERAGE_ADJUSTMENTS_PATH` or `NFL_TEAM_COVERAGE_PATH` at them, but they
  are no longer required.

During prop generation, the script scales the per-player quantiles and median
for passing/receiving markets (and anytime TD probabilities) when both a team
coverage tag and a player rule are present.
