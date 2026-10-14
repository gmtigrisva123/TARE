"""Inference, from III-C9.

The independent unit is the student. A frame is not independent of its
neighbours and folds from different seeds reuse the same students, so
neither counts as a replicate. Intervals come from a nonparametric bootstrap
that resamples students with replacement and recomputes the statistic on
each resample, carrying the group of a student along with the student.

The paired comparison of two variants on ERROR uses an exact sign-flip
permutation test on the per-student gains. With N students there are 2^N
equally likely sign patterns, so the smallest attainable one-sided p-value
is 2^-N. The change in the equalized-odds gap is not a sum over students, so
its one-sided p-value is the share of bootstrap resamples with the change at
or below zero.

When N is small the effect sizes with intervals carry the evidence and the
p-value does not. Neither a Wilcoxon test over seed-by-fold cells nor a
bootstrap over frames is used for any claim.
"""

import itertools

import numpy as np

from . import config


def bootstrap_students(students, statistic, n_boot=config.N_BOOTSTRAP, seed=0,
                       level=0.90):
    """A percentile interval for a statistic of the student set.

    `statistic` takes a list of students, possibly with repeats, and returns
    a number. Anything attached to a student, including the student's group,
    travels with them into the resample.
    """
    rng = np.random.default_rng(seed)
    students = list(students)
    draws = []
    for _ in range(n_boot):
        resample = rng.choice(students, size=len(students), replace=True).tolist()
        value = statistic(resample)
        if value is not None and np.isfinite(value):
            draws.append(float(value))

    if not draws:
        return {"point": float("nan"), "low": float("nan"), "high": float("nan"),
                "n_effective": 0}
    tail = (1.0 - level) / 2.0
    return {
        "point": float(statistic(students)),
        "low": float(np.quantile(draws, tail)),
        "high": float(np.quantile(draws, 1.0 - tail)),
        "n_effective": len(draws),
        "draws": draws,
    }


def sign_flip_test(gains, max_exact=22):
    """An exact one-sided sign-flip permutation test on the per-student gains.

    The null is that the sign of each gain is equally likely either way, so
    the reference distribution is the mean gain under all 2^N sign patterns.
    The p-value is the share of patterns whose mean is at or above the
    observed one, which tests whether the variant lowers the error.

    Enumeration is exact up to `max_exact` students, which covers the cohort
    of 16 in assumption A6. Above that the patterns are sampled, and the
    result says which path was taken.
    """
    values = np.array([g for g in gains if np.isfinite(g)], dtype=float)
    n = len(values)
    if n == 0:
        return {"p_value": float("nan"), "n_students": 0, "exact": False}

    observed = float(values.mean())
    if n <= max_exact:
        total, at_least = 0, 0
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            total += 1
            if float(np.mean(np.array(signs) * values)) >= observed:
                at_least += 1
        return {
            "p_value": at_least / total,
            "n_students": n,
            "exact": True,
            "smallest_attainable": 2.0 ** (-n),
            "observed_mean": observed,
        }

    rng = np.random.default_rng(0)
    draws = rng.choice((-1.0, 1.0), size=(100000, n))
    means = (draws * values).mean(axis=1)
    return {
        "p_value": float((means >= observed).mean()),
        "n_students": n,
        "exact": False,
        "observed_mean": observed,
    }


def gap_change_p_value(bootstrap_draws):
    """The one-sided p-value for Delta_EO of (12).

    The share of bootstrap resamples in which the change is at or below
    zero, meaning the variant did not narrow the gap.
    """
    draws = np.asarray([d for d in bootstrap_draws if np.isfinite(d)], dtype=float)
    if draws.size == 0:
        return float("nan")
    return float((draws <= 0.0).mean())


def holm(p_values, level=config.HOLM_LEVEL):
    """Holm's correction inside the primary family.

    The family of III-C9 is the comparison of TARE-S and of TARE-FULL with
    RAW on ERROR, on EO_reg and on EO_bin(2.5), the last two for each of the
    two proxies, which gives ten comparisons. Every other quantity in the
    paper is descriptive and does not enter here.

    `p_values` maps a comparison name to its p-value. The returned record
    says, for each one, the adjusted threshold it was tested against and
    whether it was rejected. Holm stops at the first failure, so comparisons
    after it are not rejected whatever their p-value.
    """
    items = sorted(p_values.items(), key=lambda kv: (np.isnan(kv[1]), kv[1]))
    m = len(items)
    out, still_rejecting = {}, True
    for rank, (name, p) in enumerate(items):
        threshold = level / (m - rank)
        rejected = bool(still_rejecting and np.isfinite(p) and p <= threshold)
        if not rejected:
            still_rejecting = False
        out[name] = {"p_value": p, "threshold": threshold, "rejected": rejected,
                     "rank": rank + 1, "family_size": m}
    return out


def equivalence(interval, margin=config.EQUIVALENCE_MARGIN):
    """Whether accuracy counts as unchanged, by III-C9.

    Accuracy is declared unchanged when the 90% bootstrap interval of the
    mean gain lies inside (-margin, margin), with the margin at 0.05 rating
    points.
    """
    low, high = interval["low"], interval["high"]
    if not (np.isfinite(low) and np.isfinite(high)):
        return {"unchanged": False, "reason": "interval not available"}
    inside = (-margin < low) and (high < margin)
    return {
        "unchanged": bool(inside),
        "margin": margin,
        "low": float(low),
        "high": float(high),
    }


def primary_family_names():
    """The ten comparisons that make up the primary family of III-C9."""
    names = []
    for variant in ("TARE-S", "TARE-FULL"):
        names.append(f"{variant}:ERROR")
        for proxy in config.PROXIES:
            names.append(f"{variant}:EO_reg:{proxy}")
            names.append(f"{variant}:EO_bin(2.5):{proxy}")
    return names
