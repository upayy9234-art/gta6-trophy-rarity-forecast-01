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
* log(earners) against log-time since launch, one line per trophy, with the
  slope shrunk toward its tier's mean slope (platinum / gold / silver /
  bronze trophies move together) and toward the global mean when a trophy
  has too little history to speak for itself.

Each forecast is anchored on the last read that actually saw the trophy
(the fitted line only supplies the slope, never the level), then extrapolated
to the target in log-time: earners_hat / tracked_hat, back to a percent.

The spread is honest rather than decorative. It combines a random-walk term
that grows with the horizon, the residual scatter of each trophy's own fit
(inflated by how far past the data the target sits), the Poisson noise of a
small earner count (a trophy four players hold is noisier than one six
hundred hold), and a floor. When the history never refreshed at all -- a
single distinct page, as in the practice cases -- no slope can be estimated,
so the median stays near the last rate with a small downward drift (a fresh
refresh dilutes rates: the rules record the first one taking the tracked base
from 638 to 3,868 while almost every rate fell) and a wide spread, because
the next refresh is then pure unknown.

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

# Spread pieces, in logit units. RW_SIGMA is the random-walk volatility per
# square-root day; SPREAD_FLOOR keeps every trophy honest about at least a
# little movement; SPREAD_CAP stops a wild fit from writing a useless band.
RW_SIGMA = 0.35
SPREAD_FLOOR = 0.25
SPREAD_CAP = 2.5
# A flat history (no refresh ever seen) gets this median shift on the logit
# scale and at least this spread: the next refresh is unobserved dilution.
FLAT_SHIFT = -0.30
FLAT_SPREAD = 0.90
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


def forecast_case(case, name):
    """One case's forecast: {"trophies": [{"id", "q": [5 numbers]}]}."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_minutes = case.get("horizon_minutes") or 0
    horizon_days = max(horizon_minutes, 0) / 1440.0

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

    # --- the tracked base: one log-linear fit over the distinct states ---
    tracked_pts = [(x, math.log(s["players_tracked"]))
                   for x, s in zip(xs, states)
                   if isinstance(s.get("players_tracked"), (int, float))
                   and s["players_tracked"] > 0]
    if len(tracked_pts) >= 2:
        _, b_tracked, rv_tracked, _, _ = linfit(
            [p[0] for p in tracked_pts], [p[1] for p in tracked_pts])
        mean_x_tracked = sum(p[0] for p in tracked_pts) / len(tracked_pts)
        sd_x_tracked = math.sqrt(
            sum((p[0] - mean_x_tracked) ** 2 for p in tracked_pts)
            / len(tracked_pts))
    else:
        b_tracked, rv_tracked = 0.0, 0.0
        mean_x_tracked, sd_x_tracked = (xs[-1] if xs else 0.0), 0.0
    n_last = history[-1].get("players_tracked")
    n_last = n_last if isinstance(n_last, (int, float)) and n_last > 0 else 1.0
    n_hat = n_last * math.exp(b_tracked * (x_target - x_last))

    # --- per-trophy earner fits, then tier priors for shrinkage ---
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
            mean_x = sum(p[0] for p in pts) / len(pts)
            sd_x = math.sqrt(sum((p[0] - mean_x) ** 2 for p in pts) / len(pts))
            se_b = slope_se(rv, sxx)
        elif len(pts) == 1:
            a, b, rv = pts[0][1], None, 0.0
            mean_x, sd_x, se_b = pts[0][0], 0.0, None
        else:
            a, b, rv = None, None, 0.0
            mean_x, sd_x, se_b = 0.0, 0.0, None
        fits.append({"pts": pts, "a": a, "b": b, "rv": rv, "se_b": se_b,
                     "mean_x": mean_x, "sd_x": sd_x,
                     "tier": tr.get("tier") or "bronze"})

    # Tier priors: each tier's mean slope and the spread of slopes inside
    # it, for precision-weighted shrinkage. A tier with too few fitted
    # slopes borrows the global mean and a default spread; with no fitted
    # slope anywhere, the tracked base's own slope is the prior.
    def slope_stats(pool):
        slopes = [f["b"] for f in pool if f["b"] is not None]
        if not slopes:
            return None, None
        mean = sum(slopes) / len(slopes)
        if len(slopes) >= 2:
            var = sum((s - mean) ** 2 for s in slopes) / (len(slopes) - 1)
            sd = math.sqrt(var)
        else:
            sd = 0.5
        return mean, min(max(sd, 0.25), 1.0)

    global_mean, global_sd = slope_stats(fits)
    if global_mean is None:
        global_mean, global_sd = b_tracked, 0.5
    tier_prior = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        mean, sd = slope_stats([f for f in fits if f["tier"] == tier])
        tier_prior[tier] = (mean if mean is not None else global_mean,
                            sd if sd is not None else global_sd)

    # --- one row per trophy ---
    rows = []
    for i, tr in enumerate(trophies):
        fit = fits[i]
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
        n_seen = seen.get("players_tracked")
        n_seen = (n_seen if isinstance(n_seen, (int, float)) and n_seen > 0
                  else n_last)
        earners_last = max(rate_last * n_seen / 100.0, 0.5)
        x_seen = log_time(hours_since(t0, parse_time(seen.get("at") or "")))

        if flat:
            # No refresh was ever observed: no slope exists to fit. Hold
            # the median near the last rate, drifted down a little for the
            # dilution the next refresh brings, and spread wide.
            median_logit = logit_pct(rate_last) + FLAT_SHIFT
            spread = FLAT_SPREAD
        else:
            # Shrink this trophy's slope toward its tier's mean slope,
            # weighted by how precisely its own slope is measured: a
            # tight fit keeps its slope, a noisy one borrows the tier's.
            prior_mean, prior_sd = tier_prior.get(
                fit["tier"], (global_mean, global_sd))
            if fit["b"] is not None and fit["se_b"] is not None:
                w_data = 1.0 / (fit["se_b"] ** 2)
                w_prior = 1.0 / (prior_sd ** 2)
                b_use = ((w_data * fit["b"] + w_prior * prior_mean)
                         / (w_data + w_prior))
            elif fit["b"] is not None:
                b_use = fit["b"]
            else:
                b_use = prior_mean
            earners_hat = earners_last * math.exp(
                b_use * (x_target - x_seen))
            median_rate = 100.0 * earners_hat / n_hat
            median_logit = logit_pct(median_rate)

            # Spread: random walk over the horizon, this trophy's fit
            # residuals (tracked base's too), inflated by how far past
            # the data the target sits, plus small-count Poisson noise.
            leverage = 1.0
            if fit["sd_x"] > 0.0 and len(fit["pts"]) >= 2:
                leverage = 1.0 + ((x_target - fit["mean_x"])
                                  / fit["sd_x"]) ** 2 / len(fit["pts"])
                leverage = min(leverage, 4.0)
            resid = math.sqrt(max(fit["rv"], 0.0) + max(rv_tracked, 0.0))
            count_noise = min(1.0 / math.sqrt(max(earners_last, 1.0)), 0.6)
            spread = math.sqrt(
                (RW_SIGMA * math.sqrt(horizon_days)) ** 2
                + (resid * math.sqrt(leverage)) ** 2
                + count_noise ** 2
                + SPREAD_FLOOR ** 2)
            spread = min(max(spread, SPREAD_FLOOR), SPREAD_CAP)

        rows.append({"id": tr.get("id"),
                     "q": quantiles_from_logit(median_logit, spread)})

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
    spread = RW_SIGMA * math.sqrt(horizon_days) if horizon_days > 0 else 0.5
    spread = max(spread, SPREAD_FLOOR)
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
