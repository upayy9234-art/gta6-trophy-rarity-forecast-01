# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/upayy9234-art/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/upayy9234-art/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one: for every PlayStation 5 trophy of Grand Theft Auto VI, five quantiles of its earn rate on Exophase at the case's target, from the reads the house made before the checkpoint.

## How it works

An earn rate is a ratio — earners over players tracked — and Exophase refreshes its page in jumps, so the history is a step function. This program fits the two sides of the ratio separately, in log-time since launch:

- **Players tracked**: one log-linear fit over the distinct pages in the history.
- **Earners, per trophy**: one log-linear fit each, with the slope shrunk toward a prior slope from the trophy's own description. Each description says roughly how many hours of play the trophy takes (a story chapter is a few hours, the platinum is the whole game); earners grow like E(t) = E0 · (1 + t/1.5)^(2.0 / H^1.25), so fast trophies gain earners quickly and slow ones barely move. The shrinkage is precision-weighted — a tight fit keeps its slope, a noisy one borrows the prior.

Each forecast is anchored on the last read that actually saw the trophy (the fit supplies only the slope, never the level) and extrapolated to the target: predicted earners over predicted tracked players, back to a percent. When the history never refreshed at all — a single distinct page, as in the practice cases — the description curve sets the level directly instead.

Round 3 adds two guards against saturating earners, the round-2 entry's weak spot: a straight line fitted over the whole history keeps pointing up even after earners stop accelerating. First, a quadratic is fitted to log(earners) alongside the line, and where it bends down the slope used is half line, half the quadratic's implied secant slope to the target — recent growth, not the whole-history average. Second, a saturating rate model, s(t) = A − B·exp(−t/τ), is fitted to the observed rates (grid over τ), and when it fits the history clearly better than the power-law forecast (residual sum of squares under 60% of it), the median moves halfway, in logit space, toward the saturating model's target rate.

The spread is 0.08 on the logit scale for most trophies, and 0.05 for a trophy whose median sits at the 0.01% floor (0.25 for the flat branch's description-only level). It was chosen by scoring the program before submit on every evaluation available: six synthetic final-like cases from round 2 (two truth families, checkpoint 59.5 h, target 168 h), twelve fresh held-out cases (new seeds, a mixed family, a weekend-boost variant, a saturating-base variant, a wrong-prior variant), a 12-hour real-data case (practice checkpoint to the live page at 15:28 UTC on launch day), and the reconstructed practice answers. Losses on the eighteen final-like cases: 0.024 on average (round 2: 0.052); on round 2's own six cases 0.020 (round 2: 0.051). The saturating family, round 2's weak spot, improves from 0.074–0.077 to 0.020–0.024. Real-data: 0.187 and 0.111 (round 2: 0.187 and 0.115). A narrower spread wins a little more on the synthetic cases, but the forecast window covers a weekend and the live round-2 entry scored about twice its synthetic average, so the spread is kept at the level that stays near-optimal when the median carries extra error, not the level that is best when the median is exactly right.

Quantiles come from a logit-normal around the median, so a forecast that is too narrow pays for every trophy outside it and one too wide pays at every trophy.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` is the whole entry: Python 3.13, standard library only, no network, no writes outside the output directory, fully deterministic — two runs on the same input are byte identical. It reads every `*.in` file in the input directory and writes one `*.out` per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
