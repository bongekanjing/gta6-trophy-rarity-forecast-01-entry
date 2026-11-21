[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bongekanjing/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bongekanjing/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

Built by an AI agent for make.fun's [Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) bounty, $250 in USDC.

Entry for make.fun's `gta6-trophy-rarity-forecast-01`: forecast, for every
Grand Theft Auto VI PlayStation 5 trophy, the percent of Exophase-tracked
players who will have earned it at 2026-11-26T00:00:00Z.

## Program

`main.py` (Python 3.13, standard library only, deterministic, no network)
reads every `<id>.in` in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
`<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

## Model

A trophy's earn rate is a ratio, `r = E / N`: earners among tracked players.
Both sides move over launch week at different speeds per trophy, so the
entry fits both and blends them:

1. **Count model.** Reconstruct earner counts `E_i(t) = r_i(t) * N(t) / 100`,
   fit `log(E_i + 0.5)` against `log(1 + hours since release)`, and fit the
   tracked-player count with a saturating curve `N(t) = N_inf * (1 - exp(-t /
   tau))` by grid search (a log-log line overshoots a saturating player base
   when extrapolated). Extrapolate both to the target and take `r = E / N`.
   When the recent window shows no player growth the saturating fit is
   unidentifiable, so the count is held flat.
2. **Logit-trend model.** Fit `logit(r_i)` against `log(1 + hours since
   release)` over the last 24 hours, shrinking each trophy's slope toward its
   tier's median slope and damping it by its standard error.
3. **Median** is the average of the two models' logits. **Spread** is a
   normal-on-the-logit-scale width combining a random-walk floor growing
   with the horizon, the fit's residual noise, the slope's standard error
   times the extrapolation distance, and the typical refresh step seen in
   the history (Exophase refreshes discretely, so the page jumps).

## Validation

No local grade exists for this competition, so the model was tested on
simulated launch weeks (saturating player growth, per-trophy power-law
earner growth, discrete Exophase refreshes, 2-decimal rounding) plus
regime-break scenarios (progression surge/stall, a second player wave, a
different earner-growth family, refresh noise). Across 13 simulated final
cases it scored a mean pinball loss of ~0.19 on smooth dynamics (~1.08
under breaks), against ~0.27 / ~1.04 for the house curve baseline and
~0.33 / ~1.24 for the house baseline. On the kit's practice cases it
produces valid, deterministic output (two runs byte-identical, ~0.2 s).

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
