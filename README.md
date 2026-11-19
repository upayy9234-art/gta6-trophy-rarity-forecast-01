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

The spread is a single number, 0.3 on the logit scale, chosen by scoring the program before submit on every evaluation available: four synthetic final-like cases (two truth families, checkpoint 59.5 h, target 168 h), a weekend-boost stress variant, a 12-hour real-data case (practice checkpoint to the live page at 15:28 UTC on launch day), and the reconstructed practice answers. Losses: synthetic finals 0.039–0.077 (round 1: 0.144–0.174), real-data 0.187 and 0.115 (round 1: 1.723 and 0.733).

Quantiles come from a logit-normal around the median, so a forecast that is too narrow pays for every trophy outside it and one too wide pays at every trophy.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` is the whole entry: Python 3.13, standard library only, no network, no writes outside the output directory, fully deterministic — two runs on the same input are byte identical. It reads every `*.in` file in the input directory and writes one `*.out` per case: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
