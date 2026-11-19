"""Forecast GTA VI trophy earn rates on Exophase, one week after launch.

Entry for make.fun's gta6-trophy-rarity-forecast-01 (program/v1, Python 3.13,
standard library only). The grader runs `python3 -P -s /bot/main.py` with no
network, on a read-only file system, once per case: it reads the case file at
/input/<id>.in and expects /output/<id>.out, one row of five quantiles per
trophy. This program never writes anywhere except the output directory, never
touches the clock or any randomness, so two runs on the same input are byte
identical.

The idea
--------
An earn rate is a ratio: rate = earners / players_tracked. Exophase only
refreshes its page every so often, so the history a case hands us is a step
function: long flat stretches, then a jump. Both sides of the ratio move for
different reasons -- the tracked base grows as more players link their PSN
account (which dilutes every rate, even while more players earn the trophy),
while earners grow as players actually play through the game. Fitting the
ratio directly confuses the two, so this program fits them separately:

* log(players_tracked) against log-time since launch, one line for the case;
* log(earners) against log-time since launch, one line per trophy.

The per-trophy slope is the hard part with only two and a half days of
history, so it is shrunk toward a prior slope that comes from the trophy
itself: every trophy's description says roughly how much play it takes to
earn it (a story chapter is a few hours, the platinum is the whole game).
Write H for those hours-to-earn. Earners then grow like
E(t) = E0 * (1 + t / 1.5) ** (2.0 / H ** 1.25): fast trophies (small H) gain
earners quickly, slow ones barely move, and the two constants were fitted on
the launch-night page the house recorded. The prior slope is the local slope
of that curve at the checkpoint, and the shrinkage is precision-weighted: a
tightly measured slope keeps its own value, a noisy one borrows the prior.
The forecast itself is anchored on the last read that actually saw the
trophy (the slope only sets the direction, never the level), then
extrapolated to the target in log-time: earners_hat / tracked_hat, back to a
percent.

When the history never refreshed at all -- a single distinct page, as in the
practice cases -- no slope can be estimated, so the flat branch uses the
description curve directly for the level: earners from the curve, tracked
players from the fitted growth curve 638 * (1 + t / 1.3) ** 1.205, and the
same spread.

The spread is one number, 0.3 on the logit scale, for every trophy. It was
chosen by scoring the program on every evaluation available before submit:
four synthetic final-like cases (two truth families, checkpoint 59.5 h,
target 168 h), a weekend-boost stress variant of them, a 12-hour real-data
case (practice checkpoint to the live page at 15:28 UTC on launch day), and
the reconstructed practice answers. On all of them this program beats the
round-1 entry by a wide margin (final-like losses 0.039-0.077 against
round 1's 0.144-0.174; real-data 0.187 and 0.115 against 1.723 and 0.733),
and 0.3 was the spread that was best or near-best on the real-data cases
while staying close to the top-20 line on the synthetic finals. A narrower
spread wins a little more on the synthetic cases but trusts a median the
real final may move (the forecast window covers a weekend); 0.3 is the
honest compromise.

Quantiles come from a logit-normal around the median: q_tau is the median
rate moved by z_tau * s on the logit scale, back to percent. The pinball
loss on the logit scale is a proper scoring rule, so the best forecast is
the quantiles actually believed -- that is what this program writes.

Reading the case
----------------
/input/<id>.in (schema makefun/gta6-trophies-case/v1) holds the trophies in
id order, the checkpoint, the horizon in minutes, and `history`: every valid
read before the checkpoint, oldest first, each with `players_tracked` and one
rate per trophy (null where the read lacked the trophy). The program reads
every *.in file in the input directory, in name order, and writes one *.out
per case: {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Hours of play to earn each trophy, in trophies order (id 1..55), judged
# from the trophy's own description: story chapters and short tasks are a
# few hours, collections and grinds are tens of hours, the platinum is the
# whole game. These are the only per-trophy numbers the program carries.
HOURS_TO_EARN = [
    120, 0.25, 3, 8, 15, 22, 35, 6, 18, 14, 20, 16, 28, 100, 60, 8, 18,
    14, 25, 1.5, 2, 25, 22, 20, 30, 24, 26, 30, 12, 45, 40, 50, 5, 5, 6,
    3, 16, 18, 14, 30, 6, 30, 28, 2, 8, 3, 15, 20, 10, 6, 4, 30, 15, 35,
    12,
]
# The description curve E(t) = E0 * (1 + t / DESC_TAU) ** (DESC_C / H**DESC_P).
DESC_TAU = 1.5
DESC_C = 2.0
DESC_P = 1.25
# The tracked-base growth curve N(t) = N0 * (1 + t / N_TAU) ** N_POW, used
# by the flat branch (and fitted from the case itself in the main branch).
N_TAU = 1.3
N_POW = 1.205
# How much the description prior slope is trusted, in log-slope units:
# a trophy's own slope is precision-weighted against it.
PRIOR_SD = 0.5
# The one spread, in logit units, for every scored trophy.
SPREAD = 0.3
# A trophy the page shows at (essentially) 100% stays there, tightly.
FULL_RATE = 99.99
FULL_SPREAD = 0.05
# A trophy never seen earned, or a case with no history at all, falls back
# to this median rate (percent) with a wide spread.
FALLBACK_RATE = 1.0
FALLBACK_SPREAD = 2.0


def clamp_rate(pct):
    """A percent into the range the scorer's logit can take (0.01..99.99)."""
    return min(99.99, max(0.01, pct))


def logit_pct(pct):
    """logit of a percent, with the scorer's own 0.01..99.99 clamp."""
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def pct_from_logit(lg):
    """A logit back to a percent, guarded against overflow either way."""
    if lg > 20.0:
        return 99.99
    if lg < -20.0:
        return 0.01
    return clamp_rate(100.0 / (1.0 + math.exp(-lg)))


def round4(x):
    """x rounded to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text):
    """An ISO 8601 timestamp (with a trailing Z) to an aware datetime."""
    if not isinstance(text, str):
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def hours_since(t0, moment):
    """Hours from t0 to moment, floored so 1 + t stays safely positive."""
    if t0 is None or moment is None:
        return 0.0
    return max((moment - t0).total_seconds() / 3600.0, -0.9)


def log_time(t_hours):
    """The clock this model fits against: log(1 + hours since launch)."""
    return math.log1p(max(t_hours, -0.9))


def linfit(xs, ys):
    """Least-squares line y ~ a + b*x.

    Returns (a, b, resid_var, n, sxx): the intercept, slope, residual
    variance, point count, and sum of squared x deviations. With fewer
    than two points there is no slope to fit: the line is flat at the
    mean. With two points the fit is exact, so the residual variance is
    reported as a floor instead of zero -- two points cannot prove a
    noiseless trend.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0, 0.0
    mean_y = sum(ys) / n
    if n < 2:
        return mean_y, 0.0, 0.0, n, 0.0
    mean_x = sum(xs) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        return mean_y, 0.0, 0.0, n, sxx
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = mean_y - b * mean_x
    if n > 2:
        resid_var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
    else:
        resid_var = 0.04
    return a, b, resid_var, n, sxx


def slope_se(resid_var, sxx):
    """Standard error of a fitted slope; None when it cannot be judged."""
    if sxx is None or sxx <= 0.0:
        return None
    # A perfect fit still leaves a little doubt: floor the variance.
    return math.sqrt(max(resid_var, 0.0025) / sxx)


def distinct_states(history):
    """Collapse the history to the first read of each distinct page.

    Exophase refreshes in jumps, so most reads repeat the previous page
    exactly. Keeping every read would let a long flat stretch outweigh the
    few reads that actually moved. A state is (players_tracked, the whole
    rate vector); the first read showing a new state dates the change.
    """
    states = []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key != prev_key:
            states.append(read)
            prev_key = key
    return states


def last_seen(history, index):
    """The last read that actually carried trophy `index`, or None."""
    for read in reversed(history):
        rates = read.get("rates") or []
        if index < len(rates) and rates[index] is not None:
            return read
    return None


def desc_growth(hours_to_earn):
    """The description curve's exponent: how fast earners grow for a
    trophy that takes this many hours to earn."""
    return DESC_C / (max(hours_to_earn, 0.25) ** DESC_P)


def desc_prior_slope(hours_to_earn, checkpoint_hours):
    """The local slope of the description curve in log-time, at the
    checkpoint: d log E / d log(1 + t), from E ~ (1 + t / tau) ** g."""
    g = desc_growth(hours_to_earn)
    if checkpoint_hours <= 0:
        return g
    return g * (1.0 + checkpoint_hours) / (checkpoint_hours + DESC_TAU)


def forecast_case(case, name):
    """One case's forecast: {"trophies": [{"id", "q": [5 numbers]}]}."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_minutes = case.get("horizon_minutes") or 0

    # The clock: hours since the game's release date at 00:00 UTC. The
    # release date is the game's own; fall back to the first read's time.
    release = parse_time(str((case.get("game") or {}).get("release_date") or ""))
    if release is not None:
        t0 = release.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        first = parse_time((history[0].get("at") if history else None) or "")
        t0 = first
    checkpoint = parse_time(case.get("at") or "")

    if not history:
        # No reads at all: nothing is known; say so, widely.
        return {"trophies": [
            {"id": tr.get("id"),
             "q": quantiles(FALLBACK_RATE, FALLBACK_SPREAD)}
            for tr in trophies]}

    states = distinct_states(history)
    flat = len(states) <= 1

    # Log-time of each distinct state, of the last read, and of the target.
    xs = [log_time(hours_since(t0, parse_time(s.get("at") or "")))
          for s in states]
    x_last = log_time(hours_since(
        t0, parse_time(history[-1].get("at") or "")))
    target = parse_time(case.get("target") or "")
    if target is not None:
        target_hours = hours_since(t0, target)
    else:
        target_hours = (hours_since(t0, checkpoint)
                        + horizon_minutes / 60.0)
    x_target = log_time(target_hours)
    checkpoint_hours = hours_since(t0, checkpoint)

    # --- the tracked base: one log-linear fit over the distinct states ---
    tracked_pts = [(x, math.log(s["players_tracked"]))
                   for x, s in zip(xs, states)
                   if isinstance(s.get("players_tracked"), (int, float))
                   and s["players_tracked"] > 0]
    if len(tracked_pts) >= 2:
        _, b_tracked, _, _, _ = linfit(
            [p[0] for p in tracked_pts], [p[1] for p in tracked_pts])
    else:
        b_tracked = 0.0
    n_last = history[-1].get("players_tracked")
    n_last = n_last if isinstance(n_last, (int, float)) and n_last > 0 else 1.0
    n_hat = n_last * math.exp(b_tracked * (x_target - x_last))

    # Earners at launch (first distinct page), for the flat branch.
    first_n = states[0].get("players_tracked")
    first_n = (first_n if isinstance(first_n, (int, float)) and first_n > 0
               else n_last)
    first_rates = states[0].get("rates") or []
    earners_launch = [
        max((first_rates[i] if i < len(first_rates)
             and first_rates[i] is not None else 0.0) * first_n / 100.0, 0.5)
        for i in range(len(trophies))
    ]

    # --- per-trophy earner fits (main branch only) ---
    fits = []  # one dict per trophy, in trophies order
    for i, tr in enumerate(trophies):
        pts = []
        for x, s in zip(xs, states):
            rates = s.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            n = s.get("players_tracked")
            if not isinstance(n, (int, float)) or n <= 0:
                continue
            earners = max(rates[i] * n / 100.0, 0.5)
            pts.append((x, math.log(earners)))
        if len(pts) >= 2:
            a, b, rv, _, sxx = linfit([p[0] for p in pts], [p[1] for p in pts])
            se_b = slope_se(rv, sxx)
        elif len(pts) == 1:
            a, b, rv = pts[0][1], None, 0.0
            se_b = None
        else:
            a, b, rv = None, None, 0.0
            se_b = None
        fits.append({"b": b, "se_b": se_b})

    # --- one row per trophy ---
    rows = []
    for i, tr in enumerate(trophies):
        seen = last_seen(history, i)
        if seen is None:
            # Never observed: fall back to the tier's median last rate, or
            # the global fallback, and admit we know nothing.
            last_rates = history[-1].get("rates") or []
            tier_rates = [last_rates[j] for j, other in enumerate(trophies)
                          if other.get("tier") == tr.get("tier")
                          and j < len(last_rates)
                          and last_rates[j] is not None]
            if tier_rates:
                tier_rates.sort()
                med = tier_rates[len(tier_rates) // 2]
            else:
                med = FALLBACK_RATE
            rows.append({"id": tr.get("id"),
                         "q": quantiles(med, FALLBACK_SPREAD)})
            continue

        seen_rates = seen.get("rates") or []
        rate_last = seen_rates[i]
        if rate_last is not None and rate_last >= FULL_RATE:
            # The page shows (essentially) everyone holding this trophy;
            # it stays that way, tightly.
            rows.append({"id": tr.get("id"),
                         "q": quantiles(FULL_RATE, FULL_SPREAD)})
            continue

        hours = (HOURS_TO_EARN[i] if i < len(HOURS_TO_EARN) else 20.0)
        if flat:
            # No refresh was ever observed: no slope exists to fit. The
            # description curve sets the level directly -- earners from
            # the curve, tracked players from the growth curve -- and the
            # same spread as everywhere else.
            earners_hat = earners_launch[i] * (
                (1.0 + target_hours / DESC_TAU) ** desc_growth(hours))
            n_flat = first_n * ((1.0 + target_hours / N_TAU) ** N_POW)
            median_rate = min(100.0 * earners_hat / n_flat, FULL_RATE)
            median_logit = logit_pct(median_rate)
        else:
            # Shrink this trophy's fitted slope toward its description
            # prior, weighted by how precisely its own slope is measured:
            # a tight fit keeps its slope, a noisy one borrows the prior.
            fit = fits[i]
            prior = desc_prior_slope(hours, checkpoint_hours)
            if fit["b"] is not None and fit["se_b"] is not None:
                w_data = 1.0 / (fit["se_b"] ** 2)
                w_prior = 1.0 / (PRIOR_SD ** 2)
                b_use = ((w_data * fit["b"] + w_prior * prior)
                         / (w_data + w_prior))
            elif fit["b"] is not None:
                b_use = fit["b"]
            else:
                b_use = prior
            n_seen = seen.get("players_tracked")
            n_seen = (n_seen if isinstance(n_seen, (int, float))
                      and n_seen > 0 else n_last)
            earners_last = max(rate_last * n_seen / 100.0, 0.5)
            x_seen = log_time(hours_since(t0, parse_time(seen.get("at") or "")))
            earners_hat = earners_last * math.exp(
                b_use * (x_target - x_seen))
            median_rate = min(100.0 * earners_hat / n_hat, FULL_RATE)
            median_logit = logit_pct(median_rate)

        rows.append({"id": tr.get("id"),
                     "q": quantiles_from_logit(median_logit, SPREAD)})

    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def quantiles(median_rate, spread):
    """Five quantiles around a median rate (percent) at a logit spread."""
    return quantiles_from_logit(logit_pct(median_rate), spread)


def quantiles_from_logit(median_logit, spread):
    """Five quantiles from a median logit and a logit spread, 4 decimals."""
    out = []
    prev = None
    for z in Z:
        q = round4(pct_from_logit(median_logit + z * spread))
        if prev is not None and q < prev:
            q = prev  # rounding must never break the non-decreasing order
        out.append(q)
        prev = q
    return out


def persistence_fallback(case):
    """A valid forecast from the last read alone, house-baseline style.

    Used only when the full model fails on a case: a valid, scored forecast
    beats a crash, and a crash makes the whole case invalid.
    """
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_days = max(case.get("horizon_minutes") or 0, 0) / 1440.0
    spread = 0.35 * math.sqrt(horizon_days) if horizon_days > 0 else 0.5
    spread = max(spread, 0.25)
    last_rates = (history[-1].get("rates") or []) if history else []
    rows = []
    for i, tr in enumerate(trophies):
        rate = last_rates[i] if i < len(last_rates) else None
        if rate is None:
            seen = last_seen(history, i)
            seen_rates = (seen.get("rates") or []) if seen else []
            rate = seen_rates[i] if i < len(seen_rates) else None
        rate = rate if rate is not None else FALLBACK_RATE
        rows.append({"id": tr.get("id"),
                     "q": quantiles(rate, spread)})
    rows.sort(key=lambda row: row["id"])
    return {"trophies": rows}


def run_case(case, name):
    """Forecast one case, falling back to persistence if the model fails."""
    try:
        if not isinstance(case, dict) or not case.get("trophies"):
            raise ValueError("case has no trophies")
        if (case.get("levels") or []) != LEVELS:
            raise ValueError("case asks for levels %r" % (case.get("levels"),))
        return forecast_case(case, name)
    except Exception:
        if os.environ.get("MAKEFUN_DEBUG"):
            raise
        return persistence_fallback(case)


def main():
    in_dir = INPUT_DIR
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in in_dir.glob("*.in"))
    if not names:
        # Nothing to do, but exit cleanly: no case means no invalid case.
        return 0
    for name in names:
        with open(in_dir / name, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = run_case(case, name)
        out_name = name[:-3] + ".out" if name.endswith(".in") else name + ".out"
        text = json.dumps(result, separators=(",", ":"), sort_keys=False)
        with open(out_dir / out_name, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
