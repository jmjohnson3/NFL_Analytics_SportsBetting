# NFL Analytics Sports Betting

## Why to Hold Off for Now

The current pipeline still leans on synthetic data to pad historical records. When the
system cannot find player outcomes it fabricates placeholder "prior" rows with
`is_synthetic=True` so that training splits and ROI summaries do not break. That keeps
the scripts running but it also means backtests can be built on assumptions instead of
real bookmaker prices or verified player results, which is a red flag for live betting.

Several situational features are also unpopulated. The team-strength feature builder
initializes travel, rest, weather, and even average timezone adjustments to `NaN`, so
every scoring model is trained without that context. The downstream imputer will keep
warning about those missing columns until actual values are collected and stored.

Finally, ROI backtests have started to fail whenever completed games are missing
moneyline history. The evaluation routine now refuses to produce profitability metrics
and instructs you to backfill historical odds first. That is exactly the state of the
repository today—the pipeline cannot certify an edge without those prices, and neither
should you.

## Prerequisites Before Betting Real Money

1. **Backfill real sportsbook odds** for all historical games that fall in your
   training and validation windows so each completed matchup has recorded moneylines.
2. **Populate rest, travel, timezone, and weather inputs** (or drop the features) so
   the feature warnings disappear because the data exist, not because they are masked.
3. **Paper-trade future slates** once the data gaps above are closed and confirm that
   the live ROI remains positive against the true lines you intend to bet.

Only after these gaps are addressed should you consider deploying the model in a live
betting environment.
