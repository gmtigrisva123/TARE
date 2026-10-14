"""The four audit measures of III-C6.

ERROR is accuracy, the mean absolute error. GAP reads fairness through
equalized odds in a regression form and a binary form that both condition on
the true rating, together with the error disparity across students and a
rank association between error and a continuous proxy. LEAK asks whether an
auxiliary model can recover that proxy from the features.

The unit of analysis is the student. Seeds change only how students are
partitioned, so they are averaged over inside each student first, by (8).
"""

from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config


# --- (8) to (10), ERROR ---------------------------------------------------

def frame_errors(predictions, frames):
    """epsilon_i of (8), the absolute error averaged over seeds."""
    targets = np.array([f.target for f in frames])
    return np.nanmean(np.abs(predictions - targets[None, :]), axis=0)


def student_errors(frame_error, frames, students):
    """e_s of (8), the mean over the test frames of a student.

    I_s includes the frames with u_i = 0, because they are scored.
    """
    out = {}
    student_of = np.array([f.student for f in frames])
    for student in students:
        rows = student_of == student
        out[student] = float(np.nanmean(frame_error[rows])) if rows.any() else float("nan")
    return out


def error_summary(student_error, raw_student_error=None, with_intervals=True):
    """The quantities of (9) and (10).

    Delta is e_s^RAW minus e_s^v, so a positive value means the variant has
    the lower error. e_max and e_spread of (10) do not refer to apparent
    scale; the criterion that does is equalized odds.
    """
    students = sorted(student_error)
    values = np.array([student_error[s] for s in students])
    summary = {
        "mean_error": float(np.nanmean(values)),
        "e_max": float(np.nanmax(values)),
        "e_spread": float(np.nanmax(values) - np.nanmin(values)),
    }
    if raw_student_error is not None:
        gains = {s: raw_student_error[s] - student_error[s] for s in students}
        summary["delta_per_student"] = gains
        summary["mean_delta"] = float(np.nanmean(list(gains.values())))

    if with_intervals:
        summary["intervals"] = _error_intervals(student_error)
    return summary


def _error_intervals(student_error):
    """Bootstrap intervals for the quantities of (9) and (10).

    Section III-C9 draws the intervals for (10) from a resample of students,
    so each statistic is recomputed on whoever was drawn.
    """
    from . import stats

    students = sorted(student_error)

    def reduce_with(function):
        def statistic(resample):
            values = [student_error[s] for s in resample]
            values = [v for v in values if np.isfinite(v)]
            return function(values) if values else float("nan")
        return stats.bootstrap_students(students, statistic)

    drop_draws = lambda d: {k: v for k, v in d.items() if k != "draws"}
    return {
        "mean_error": drop_draws(reduce_with(np.mean)),
        "e_max": drop_draws(reduce_with(np.max)),
        "e_spread": drop_draws(reduce_with(lambda v: np.max(v) - np.min(v))),
    }


def error_by_activity(frame_error, frames, students):
    """e-bar per activity, by restricting (8) to the frames of that activity."""
    out = {}
    for activity in config.ACTIVITIES:
        subset = [i for i, f in enumerate(frames) if f.activity == activity]
        if not subset:
            continue
        per_student = {}
        for student in students:
            rows = [i for i in subset if frames[i].student == student]
            if rows:
                per_student[student] = float(np.nanmean(frame_error[rows]))
        out[activity] = float(np.nanmean(list(per_student.values())))
    return out


# --- (14), apparent-scale proxies ----------------------------------------

def apparent_scale_proxies(frames, students):
    """P_iod and P_sh of (14).

    Both are medians over the usable lecture frames of a student, where
    students sit still and face forward. Assumption A5 applies: these are
    computed from images and are not body height, so any statement about
    groups refers to apparent scale.
    """
    iod, shoulder = {}, {}
    for student in students:
        rows = [
            f.landmarks for f in frames
            if f.student == student
            and f.activity == config.LECTURE_ACTIVITY
            and f.landmarks.usable
        ]
        if not rows:
            continue
        iod[student] = float(np.median([r.iod for r in rows]))
        shoulder[student] = float(np.median([r.shoulder_width for r in rows]))
    return {"P_iod": iod, "P_sh": shoulder}


def scale_groups(proxy):
    """G(s) of III-C6c, the tertile of the proxy among the students who have it.

    n_g = 3: small, medium and large apparent scale. With N_lec = 16 each
    group holds five or six students.
    """
    students = sorted(proxy, key=lambda s: proxy[s])
    n = len(students)
    groups = {}
    for rank, student in enumerate(students):
        groups[student] = min(int(rank * config.N_GROUPS / n) + 1, config.N_GROUPS)
    return groups


# --- (11) and (12), equalized odds ---------------------------------------

def _label_class(target):
    for q, (low, high) in enumerate(config.LABEL_CLASSES, start=1):
        if (low <= target <= high) if q == 1 else (low < target <= high):
            return q
    return None


def equalized_odds_cells(frames, groups):
    """Which frames and which students sit in each (class, group) cell.

    Cell membership depends only on the label and the group, so the same
    cells enter every variant, which is what III-C6c requires.
    """
    cells = {}
    for idx, frame in enumerate(frames):
        group = groups.get(frame.student)
        if group is None:
            continue
        q = _label_class(frame.target)
        if q is None:
            continue
        cell = cells.setdefault((q, group), {"rows": [], "students": set()})
        cell["rows"].append(idx)
        cell["students"].add(frame.student)
    return cells


def equalized_odds(statistic, cells):
    """EO of (11), the largest gap between two groups inside one label class.

    A cell holding fewer than three students is excluded from the maxima and
    listed. Cell means, cell sizes in frames and in students, and the
    excluded cells are all returned, because III-C6c asks for the rates to
    be reported next to the gaps.
    """
    means, sizes, excluded = {}, {}, []
    for (q, group), cell in cells.items():
        rows = cell["rows"]
        value = float(np.nanmean(statistic[rows])) if rows else float("nan")
        n_students = len(cell["students"])
        sizes[(q, group)] = {"frames": len(rows), "students": n_students}
        if n_students < config.MIN_CELL_STUDENTS:
            excluded.append((q, group))
            continue
        means[(q, group)] = value

    gap = 0.0
    worst = None
    for q in sorted({q for q, _ in means}):
        present = {g: means[(q, g)] for (qq, g) in means if qq == q}
        values = [v for v in present.values() if np.isfinite(v)]
        if len(values) < 2:
            continue
        spread = max(values) - min(values)
        if spread > gap:
            gap = spread
            worst = q
    return {
        "gap": float(gap),
        "worst_class": worst,
        "cell_means": means,
        "cell_sizes": sizes,
        "excluded_cells": excluded,
    }


def gap_interval(frames, students, gap_of_rows):
    """A bootstrap interval for an equalized-odds gap of (11).

    `gap_of_rows` takes a list of frame indices and returns the gap on them.
    A resample draws students with replacement and a student's frames are
    repeated as often as the student is, and the student's group travels
    with them, which is what III-C9 requires.
    """
    from . import stats

    student_of = np.array([f.student for f in frames])
    rows_of = {s: np.flatnonzero(student_of == s).tolist() for s in students}

    def statistic(resample):
        rows = []
        for student in resample:
            rows.extend(rows_of.get(student, []))
        return gap_of_rows(rows) if rows else float("nan")

    interval = stats.bootstrap_students(students, statistic)
    return {k: v for k, v in interval.items() if k != "draws"}


def binary_statistic(predictions, threshold):
    """zeta_i of III-C6c, the exceedance indicator averaged over seeds."""
    return np.nanmean((predictions > threshold).astype(float), axis=0)


def equalized_odds_binary(predictions, frames, groups, threshold):
    """EO_bin(tau) of III-C6c.

    With the two classes Y_0 = {y <= tau} and Y_1 = {y > tau}, the cell mean
    in class 1 is the true-positive rate of the group and in class 0 the
    false-positive rate, so the gap is the larger of the two. Naming the
    other class positive exchanges the two kinds of gap and leaves the value
    unchanged.

    A threshold the predictor seldom crosses drives all rates toward zero,
    so the rates are reported beside the gap.
    """
    cells = {}
    for idx, frame in enumerate(frames):
        group = groups.get(frame.student)
        if group is None:
            continue
        q = 1 if frame.target > threshold else 0
        cell = cells.setdefault((q, group), {"rows": [], "students": set()})
        cell["rows"].append(idx)
        cell["students"].add(frame.student)

    statistic = binary_statistic(predictions, threshold)
    result = equalized_odds(statistic, cells)
    result["true_positive_rates"] = {
        g: v for (q, g), v in result["cell_means"].items() if q == 1
    }
    result["false_positive_rates"] = {
        g: v for (q, g), v in result["cell_means"].items() if q == 0
    }
    result["threshold"] = threshold
    return result


def equalized_odds_regression(frame_error, frames, groups):
    """EO_reg of III-C6c, the conditional mean absolute error across groups.

    Uses the absolute error epsilon_i and the Q = 3 label classes of
    Table III.
    """
    cells = equalized_odds_cells(frames, groups)
    return equalized_odds(frame_error, cells)


def gap_change(raw_gap, variant_gap):
    """Delta_EO of (12). A positive value means the variant narrows the gap."""
    return float(raw_gap - variant_gap)


# --- (13), rank association ----------------------------------------------

def rank_association(student_error, proxy):
    """rho of (13), Spearman between the student error and the proxy.

    Grouping discards information when students are few, and this measure
    needs no groups.
    """
    students = sorted(set(student_error) & set(proxy))
    if len(students) < 3:
        return float("nan")
    errors = [student_error[s] for s in students]
    values = [proxy[s] for s in students]
    return float(spearmanr(errors, values).statistic)


def rank_association_interval(student_error, proxy):
    """A bootstrap interval for rho of (13), resampling students."""
    from . import stats

    students = sorted(set(student_error) & set(proxy))

    def statistic(resample):
        kept = [s for s in resample]
        errors = [student_error[s] for s in kept]
        values = [proxy[s] for s in kept]
        if len(set(values)) < 3:
            return float("nan")
        return float(spearmanr(errors, values).statistic)

    interval = stats.bootstrap_students(students, statistic)
    return {k: v for k, v in interval.items() if k != "draws"}


def association_change(raw_rho, variant_rho):
    """Delta_rho of (13). Positive means the variant weakens the association."""
    return float(abs(raw_rho) - abs(variant_rho))


# --- (15) and (16), LEAK -------------------------------------------------

def leakage(features, frames, proxy, seeds=config.SEEDS):
    """R2 of (15) and Xi of (16), on the usable lecture frames.

    Folds are leave-students-out with K_lec = min(7, N_lec) and the same
    three seeds. Both quantities are descriptive and are not used for
    hypothesis tests: the proxy takes only N_lec distinct values, and when
    N_lec is small the model has to extrapolate a constant to an unseen
    student.

    Each proxy is a median of a raw pixel feature, so RAW carries it by
    construction, and because (6) removes each student's median a low
    leakage under TARE-FULL is expected by construction for any component
    that is constant within a student. The informative comparison is
    between variants.
    """
    rows = [
        i for i, f in enumerate(frames)
        if f.activity == config.LECTURE_ACTIVITY
        and f.landmarks.usable
        and f.student in proxy
    ]
    if not rows:
        return {"r2": float("nan"), "xi": float("nan"), "n_frames": 0}

    x = features[rows]
    student_of = [frames[i].student for i in rows]
    y = np.array([proxy[s] for s in student_of])
    students = sorted(set(student_of))

    predictions = np.full((len(seeds), len(rows)), np.nan)
    n_folds = min(config.LEAK_MAX_FOLDS, len(students))
    for seed_row, seed in enumerate(seeds):
        order = list(students)
        np.random.default_rng(seed).shuffle(order)
        assignment = {s: j % n_folds for j, s in enumerate(order)}
        fold_of = np.array([assignment[s] for s in student_of])
        for fold in range(n_folds):
            test = fold_of == fold
            train = ~test
            if not test.any() or not train.any():
                continue
            model = HistGradientBoostingRegressor(
                max_iter=config.LEAK_GB_MAX_ITER,
                learning_rate=config.LEAK_GB_LEARNING_RATE,
                random_state=0,
            )
            model.fit(x[train], y[train])
            predictions[seed_row, test] = model.predict(x[test])

    pooled = np.nanmean(predictions, axis=0)
    residual = np.nansum((y - pooled) ** 2)
    total = np.nansum((y - y.mean()) ** 2)
    r2 = float(1.0 - residual / total) if total > 0 else float("nan")

    associations = []
    for column in range(x.shape[1]):
        values = x[:, column]
        finite = np.isfinite(values)
        if finite.sum() < 3 or np.allclose(values[finite], values[finite][0]):
            continue
        associations.append(abs(spearmanr(values[finite], y[finite]).statistic))
    xi = float(np.nanmean(associations)) if associations else float("nan")

    return {
        "r2": r2,
        "xi": xi,
        "n_frames": len(rows),
        "n_students": len(students),
        "intervals": _leakage_intervals(x, y, pooled, student_of, students),
    }


def _leakage_intervals(x, y, pooled, student_of, students):
    """Bootstrap intervals for (15) and (16), resampling students.

    The out-of-fold predictions already in hand are reused rather than
    refitted, so the interval reflects which students were drawn and not a
    second round of model fitting. Both quantities stay descriptive, by
    III-C6e.
    """
    from . import stats

    student_of = np.asarray(student_of)
    rows_of = {s: np.flatnonzero(student_of == s) for s in students}

    def rows_for(resample):
        picked = [rows_of[s] for s in resample if len(rows_of[s])]
        return np.concatenate(picked) if picked else np.array([], dtype=int)

    def r2_statistic(resample):
        rows = rows_for(resample)
        if rows.size == 0:
            return float("nan")
        truth, prediction = y[rows], pooled[rows]
        total = np.nansum((truth - truth.mean()) ** 2)
        if total <= 0:
            return float("nan")
        return float(1.0 - np.nansum((truth - prediction) ** 2) / total)

    def xi_statistic(resample):
        rows = rows_for(resample)
        if rows.size == 0:
            return float("nan")
        truth = y[rows]
        values = []
        for column in range(x.shape[1]):
            series = x[rows, column]
            finite = np.isfinite(series)
            if finite.sum() < 3 or np.allclose(series[finite], series[finite][0]):
                continue
            values.append(abs(spearmanr(series[finite], truth[finite]).statistic))
        return float(np.nanmean(values)) if values else float("nan")

    drop = lambda d: {k: v for k, v in d.items() if k != "draws"}
    return {
        "r2": drop(stats.bootstrap_students(students, r2_statistic)),
        "xi": drop(stats.bootstrap_students(students, xi_statistic)),
    }
