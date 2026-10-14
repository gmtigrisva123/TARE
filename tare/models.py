"""Regressors, folds and reference predictors, from III-C5.

Both regressor families are fitted with identical settings for every
variant, so a difference between variants comes from the features and not
from tuning. Ridge gets median imputation and standardization first;
histogram gradient boosting handles missing values itself. Predictions are
clipped to [1, 5].

Folds are formed over students. For each seed the student order is shuffled
and the student in position j goes to fold j mod K with K = min(4, N), so
all frames of a student are in the training set or the test set and never in
both. Frames 1.5 s apart are nearly identical, so a frame-level split would
let a model recognise the student and inflate every figure.

A frame with u_i = 0 is not discarded. Its features are entirely missing, so
it is left out of fitting, and it receives the training mean as its
prediction in every variant. Every variant is then scored on the same
frames.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config


def make_regressor(kind):
    """One of the two families of III-C5, with the settings of Table III."""
    if kind == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=config.RIDGE_ALPHA),
        )
    if kind == "gb":
        return HistGradientBoostingRegressor(
            max_iter=config.GB_MAX_ITER,
            learning_rate=config.GB_LEARNING_RATE,
            max_leaf_nodes=config.GB_MAX_LEAF_NODES,
            l2_regularization=config.GB_L2,
            random_state=0,
        )
    raise ValueError(f"Unknown regressor {kind!r}; expected 'ridge' or 'gb'.")


def clip(predictions):
    low, high = config.PREDICTION_RANGE
    return np.clip(predictions, low, high)


def student_folds(students, seed):
    """Students to folds, by the rule of III-C5.

    The order is shuffled with the seed, then position j goes to fold
    j mod K. K is min(4, N).
    """
    order = list(students)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    n_folds = min(config.MAX_FOLDS, len(order))
    assignment = {student: j % n_folds for j, student in enumerate(order)}
    return assignment, n_folds


def out_of_fold_predictions(features, frames, students, regressor_kind):
    """One out-of-fold prediction per frame per seed.

    Returns an array of shape (n_seeds, n_frames). Rows follow
    `config.SEEDS` in order.
    """
    targets = np.array([f.target for f in frames])
    usable = np.array([f.landmarks.usable for f in frames])
    student_of = np.array([f.student for f in frames])

    predictions = np.full((len(config.SEEDS), len(frames)), np.nan)
    for row, seed in enumerate(config.SEEDS):
        assignment, n_folds = student_folds(students, seed)
        fold_of = np.array([assignment[s] for s in student_of])
        for fold in range(n_folds):
            test = fold_of == fold
            train = ~test
            if not test.any() or not train.any():
                continue

            training_mean = float(targets[train].mean())
            fit_rows = train & usable
            model = make_regressor(regressor_kind)
            model.fit(features[fit_rows], targets[fit_rows])

            predict_rows = test & usable
            if predict_rows.any():
                predictions[row, predict_rows] = clip(model.predict(features[predict_rows]))
            predictions[row, test & ~usable] = training_mean
    return predictions


def apply_to_perturbed(features_clean, features_perturbed, frames, students,
                       regressor_kind):
    """Score perturbed held-out frames with predictors trained on clean data.

    Used by both ZOOM protocols, which share one arrangement: a predictor
    trained on the clean features of the training students is applied to
    features recomputed from the perturbed frames of the held-out students.

    Returns the clean and the perturbed out-of-fold predictions, both of
    shape (n_seeds, n_frames), with NaN wherever a frame was not held out or
    was not usable.
    """
    targets = np.array([f.target for f in frames])
    usable = np.array([f.landmarks.usable for f in frames])
    student_of = np.array([f.student for f in frames])

    clean = np.full((len(config.SEEDS), len(frames)), np.nan)
    perturbed = np.full((len(config.SEEDS), len(frames)), np.nan)

    for row, seed in enumerate(config.SEEDS):
        assignment, n_folds = student_folds(students, seed)
        fold_of = np.array([assignment[s] for s in student_of])
        for fold in range(n_folds):
            test = fold_of == fold
            train = ~test
            if not test.any() or not train.any():
                continue

            fit_rows = train & usable
            model = make_regressor(regressor_kind)
            model.fit(features_clean[fit_rows], targets[fit_rows])

            held_out = test & usable
            if not held_out.any():
                continue
            clean[row, held_out] = clip(model.predict(features_clean[held_out]))

            finite = held_out & np.isfinite(features_perturbed).all(axis=1)
            if finite.any():
                perturbed[row, finite] = clip(model.predict(features_perturbed[finite]))
    return clean, perturbed


def reference_predictors(frames, students, expert_ratings):
    """B1 to B4 of (7), as frame-level predictions where they have them.

    B1 predicts the training mean. B2 predicts the training mean of the
    frame's activity. B3 predicts the test mean of the student's session; it
    uses test labels and marks a ceiling, not a competitor. B4 is the error
    of one expert against the mean of the other three, a single number
    rather than a prediction, so it is returned separately.

    A model is informative only when it beats B2 and approaches B3.
    """
    targets = np.array([f.target for f in frames])
    student_of = np.array([f.student for f in frames])
    activity_of = np.array([f.activity for f in frames])

    out = {name: np.full((len(config.SEEDS), len(frames)), np.nan)
           for name in ("B1", "B2", "B3")}

    for row, seed in enumerate(config.SEEDS):
        assignment, n_folds = student_folds(students, seed)
        fold_of = np.array([assignment[s] for s in student_of])
        for fold in range(n_folds):
            test = fold_of == fold
            train = ~test
            if not test.any() or not train.any():
                continue

            out["B1"][row, test] = float(targets[train].mean())

            for activity in np.unique(activity_of[test]):
                rows = test & (activity_of == activity)
                same = train & (activity_of == activity)
                out["B2"][row, rows] = float(
                    targets[same].mean() if same.any() else targets[train].mean()
                )

            for student in np.unique(student_of[test]):
                for activity in np.unique(activity_of[test & (student_of == student)]):
                    rows = test & (student_of == student) & (activity_of == activity)
                    out["B3"][row, rows] = float(targets[rows].mean())

    out["B4"] = human_reference(frames, expert_ratings)
    return out


def human_reference(frames, expert_ratings):
    """B4 of (7): one expert against the mean of the other three.

    Averaged over the four experts and over the scored frames I.
    """
    n_experts = None
    per_expert = None
    for frame in frames:
        series = expert_ratings.get(frame.session)
        if series is None:
            continue
        second = frame.second
        values = [s[second] for s in series if second in s]
        if n_experts is None:
            n_experts = len(series)
            per_expert = [[] for _ in range(n_experts)]
        if len(values) != n_experts:
            continue
        for r in range(n_experts):
            others = [values[k] for k in range(n_experts) if k != r]
            per_expert[r].append(abs(values[r] - float(np.mean(others))))

    if not per_expert or any(len(v) == 0 for v in per_expert):
        return float("nan")
    return float(np.mean([np.mean(v) for v in per_expert]))
