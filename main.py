"""gta6-trophy-rarity-forecast-01: forecast Exophase GTA VI PS5 trophy earn rates.

Model (standard library only, deterministic, no network):

The earn rate of a trophy is a ratio, r = E / N: E earners among N tracked
players.  Both move over launch week -- N grows as more players sync, E grows
as players progress -- and the two move at different speeds per trophy, so
the rate of an early-story trophy rises while a late-game trophy's rate can
fall even as its earners grow.  Forecasting the ratio directly on the logit
scale (the house curve baseline) misses that structure, so this entry fits
both sides and blends them:

1.  Count model: reconstruct earner counts E_i(t) = r_i(t) * N(t) / 100 from
    each read, fit log(E_i + 0.5) against log(1 + hours since release), and
    fit the tracked-player count with a saturating curve
    N(t) = N_inf * (1 - exp(-t / tau)) by grid search (a log-log line
    overshoots a saturating player base when extrapolated); extrapolate
    both to the target and take r = E / N.
2.  Logit-trend model: fit logit(r_i) against log(1 + hours since release)
    over the last 24 hours, with each trophy's slope shrunk toward its
    tier's median slope (trophies of a tier share progression dynamics) and
    damped by its standard error.
3.  The median is the average of the two models' logits.  The spread is a
    normal-on-the-logit-scale width combining a random-walk floor that grows
    with the horizon, the fit's residual noise, the slope's standard error
    times the extrapolation distance, and the typical refresh step seen in
    the history (Exophase refreshes discretely, so the page jumps).

Reads are used as recorded: between refreshes the page is frozen, so a long
plateau is one observation repeated, and the fits use every read but shrink
toward the tier when a trophy's own series barely moves.

Run like the starter: `MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out
python3 main.py` reads every <id>.in and writes <id>.out.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import. (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Recent window for the fits, in hours (the house curve baseline uses 24;
# a longer window lags when the player base's growth is saturating).
WINDOW_HOURS = 24.0
# Random-walk spread, in logit units per square-root day. 0.35 balances the
# smooth launch-week dynamics (where a narrower spread wins) against a
# regime break after the checkpoint (where only width saves coverage).
SIGMA_RW = 0.35
# Floor on the predictive spread, in logit units: a refresh can move a rate
# even when the history so far is perfectly flat (the practice cases).
SPREAD_FLOOR = 0.30
# Cap on the predictive spread: beyond this the quantiles carry no signal.
SPREAD_CAP = 2.5
# Damping on the extrapolated move: 1.0 trusts the trend fully, lower pulls
# the median back toward the last read (insurance against a regime break
# after the checkpoint, which no history can show).
MOVE_DAMP = 1.0
# Weight of the count model in the median blend (rest: logit-trend model).
BLEND_COUNT = 0.5
# Weight (in reads) of the tier-median slope when shrinking a trophy's slope.
SHRINK_W = 3.0
# Weight (in reads) of the tier-median slope when shrinking a trophy's slope.
SHRINK_W = 3.0
# Multipliers on the spread's residual-noise and refresh-step terms.
RESID_MULT = 1.5
STEP_MULT = 0.8


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def logistic(lg: float) -> float:
    lg = max(-20.0, min(20.0, lg))
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def ols(xs, ys):
    """Least squares y = a + b*x. Returns (a, b, resid_std, se_b, x_mean)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0, 0.0, mx
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(n - 2, 1)
    var = sum(r * r for r in resid) / dof
    se_b = math.sqrt(var / sxx)
    return a, b, math.sqrt(var), se_b, mx


def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n_tr = len(trophies)

    # Clock: hours since the game's release (midnight UTC on release_date).
    release = case.get("game", {}).get("release_date", "2026-11-19")
    try:
        rel = datetime.fromisoformat(release).replace(tzinfo=timezone.utc)
    except ValueError:
        rel = parse_ts(history[0]["at"])
    target = parse_ts(case["target"])
    t_target = (target - rel).total_seconds() / 3600.0
    x_target = math.log1p(max(t_target, 0.0))

    # Reads: (x = log1p(hours since release), hours, players, rates).
    reads = []
    for r in history:
        t = (parse_ts(r["at"]) - rel).total_seconds() / 3600.0
        reads.append((math.log1p(max(t, 0.0)), t, r["players_tracked"], r["rates"]))
    t_last = reads[-1][1]
    x_last = reads[-1][0]

    # Recent window; fall back to all reads when the window is too thin.
    window = [rd for rd in reads if rd[1] >= t_last - WINDOW_HOURS]
    if len(window) < 3:
        window = reads
    xs_w = [rd[0] for rd in window]

    # --- players-tracked model: N saturates (new players sync fast at
    # launch, then the pool dries up), so fit N(t) = N_inf * (1 - exp(-t/tau))
    # by grid search on log N, not a log-log line (which overshoots a
    # saturating curve when extrapolated). ---
    n_last = reads[-1][2]
    best = None
    win_t = [rd[1] for rd in window]
    win_n = [max(rd[2], 1) for rd in window]
    for tau in (6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0, 768.0):
        for mult in (1.05, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0):
            n_inf = n_last * mult
            sse = 0.0
            for t, n in zip(win_t, win_n):
                pred = n_inf * (1.0 - math.exp(-(t + 0.5) / tau))
                d = math.log(max(pred, 1.0)) - math.log(n)
                sse += d * d
            if best is None or sse < best[0]:
                best = (sse, n_inf, tau)
    _, n_inf_fit, tau_fit = best
    n_target = n_inf_fit * (1.0 - math.exp(-(t_target + 0.5) / tau_fit))
    n_target = max(float(n_last), n_target)
    # When the window shows no player growth the saturating fit is
    # unidentifiable (any curve through a flat line extrapolates anywhere):
    # hold the count flat instead of inventing growth.
    if max(win_n) / min(win_n) < 1.05:
        n_target = float(n_last)

    # --- per-trophy fits (first pass, for the tier-median slopes). ---
    logit_fits = []  # (pred_logit, resid_std, se_b, dx, step_med, last_rate, n_distinct)
    count_preds = []  # predicted rate from the count model, or None
    for i in range(n_tr):
        pts = [(rd[0], rd[3][i]) for rd in window if rd[3][i] is not None]
        xs = [p[0] for p in pts]
        if not pts:
            logit_fits.append(None)
            count_preds.append(None)
            continue
        ys = [logit(p[1]) for p in pts]
        a, b, resid, se_b, mx = ols(xs, ys)
        # Typical refresh step: median |logit change| between consecutive reads.
        steps = [abs(ys[k + 1] - ys[k]) for k in range(len(ys) - 1) if xs[k + 1] > xs[k]]
        step_med = median(steps) if steps else 0.0
        n_distinct = len(set(xs))
        logit_fits.append((a, b, resid, se_b, mx, step_med, pts[-1][1], n_distinct))
        # Count model for this trophy.
        epts = [(rd[0], (rd[3][i] if rd[3][i] is not None else 0.0) * max(rd[2], 1) / 100.0)
                for rd in window if rd[3][i] is not None]
        exs = [p[0] for p in epts]
        eys = [math.log(p[1] + 0.5) for p in epts]
        if len(set(eys)) >= 3 and len(set(exs)) >= 3:
            ae, be, _, _, _ = ols(exs, eys)
            e_target = math.exp(ae + be * x_target) - 0.5
            e_target = max(0.0, e_target)
            count_preds.append(100.0 * e_target / n_target)
        else:
            # No movement to fit: earners stay where they are.
            e_last = epts[-1][1]
            count_preds.append(100.0 * e_last / n_target)

    # Tier-median logit slopes, for shrinkage.
    tier_slopes = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        slopes = [f[1] for f, tr in zip(logit_fits, trophies)
                  if f is not None and tr["tier"] == tier]
        tier_slopes[tier] = median(slopes) if slopes else 0.0
    global_slope = median([f[1] for f in logit_fits if f is not None])

    horizon_days = case["horizon_minutes"] / 1440.0
    rw = SIGMA_RW * math.sqrt(horizon_days)
    dx = x_target - x_last

    rows = []
    for i, trophy in enumerate(trophies):
        fit = logit_fits[i]
        last_rate = reads[-1][3][i]
        if last_rate is None:
            last_rate = 0.0
        if fit is None:
            # Trophy never observed: flat at the floor with a wide spread.
            med_logit = logit(0.01)
            spread = SPREAD_CAP
        else:
            a, b, resid, se_b, mx, step_med, _, n_distinct = fit
            # Shrink the slope toward the tier median (weight 3 reads' worth),
            # then damp it by its own standard error.
            tier_b = tier_slopes.get(trophy["tier"], global_slope)
            b_shrunk = (n_distinct * b + SHRINK_W * tier_b) / (n_distinct + SHRINK_W)
            damp = b_shrunk * b_shrunk / (b_shrunk * b_shrunk + se_b * se_b) if (b_shrunk or se_b) else 0.0
            b_shrunk *= damp
            y_trend = (a + b * mx) + b_shrunk * (x_target - mx)
            # Blend with the count model in logit space, then damp the
            # extrapolated move toward the last read.
            r_count = count_preds[i]
            if r_count is None:
                med_logit = y_trend
            else:
                med_logit = (1.0 - BLEND_COUNT) * y_trend + BLEND_COUNT * logit(r_count)
            if MOVE_DAMP != 1.0:
                med_logit = logit(last_rate) + MOVE_DAMP * (med_logit - logit(last_rate))
            # Spread: random walk + residual noise + slope uncertainty +
            # refresh steps, with a floor so a flat history is not trusted.
            resid_term = RESID_MULT * max(resid, 0.05)
            slope_term = se_b * max(dx, 0.0)
            step_term = STEP_MULT * step_med * math.sqrt(horizon_days)
            zero_term = 0.5 if (reads[-1][3][i] or 0.0) <= 0.0 else 0.0
            spread = math.sqrt(rw * rw + resid_term ** 2 + slope_term ** 2
                               + step_term ** 2 + zero_term ** 2 + SPREAD_FLOOR ** 2)
            spread = min(max(spread, SPREAD_FLOOR), SPREAD_CAP)
        qs = [round4(clamp_pct(logistic(med_logit + z * spread))) for z in Z]
        # Enforce non-decreasing (rounding can only tie, never invert, but be safe).
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
