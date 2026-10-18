"""The two controlled protocols of III-C7 and III-C8.

Section III-C7 varies apparent scale at the keypoint level with the map (17)
and at the image level with the matching affine warp, re-extracting
landmarks in the second case. Section III-C8 then gives every student an
independent zoom factor, which is what separates apparent scale from
stature, seating distance, camera placement and lens geometry: the factor is
unrelated to the student by construction.

Remark 3 makes the keypoint-level version a unit test. Applying (17) to all
frames of a student, the calibration frames included, gives zero drift for
TARE-S and TARE-FULL by Proposition 1, so only D^RAW is informative there.
Equations (20) and (21) are therefore computed on re-extracted landmarks
only, by Remark 4.
"""

import numpy as np

from . import config, features, models


# --- (17), the map -------------------------------------------------------

def image_center(width=config.IMAGE_WIDTH, height=config.IMAGE_HEIGHT):
    """o of (17)."""
    return np.array([width / 2.0, height / 2.0])


def apply_keypoint_map(landmarks, c, delta, width=config.IMAGE_WIDTH,
                       height=config.IMAGE_HEIGHT):
    """T_{c,delta} of (17) applied to z_i.

    The map acts on all keypoints and on the lengths w_i, IOD_i and h_i, and
    leaves the head angles unchanged. A length picks up the factor c because
    the translation cancels in a difference.
    """
    from .landmarks import RawLandmarks

    if not landmarks.usable:
        return RawLandmarks.missing()

    o = image_center(width, height)
    shifted = c * (landmarks.keypoints - o) + o + np.array([0.0, delta])
    return RawLandmarks(
        keypoints=shifted,
        shoulder_width=c * landmarks.shoulder_width,
        iod=c * landmarks.iod,
        face_height=c * landmarks.face_height,
        head_angles=landmarks.head_angles.copy(),
        usable=True,
    )


def affine_matrix(c, delta, width=config.IMAGE_WIDTH, height=config.IMAGE_HEIGHT):
    """The 2 by 3 matrix of III-C7 for the image-level version.

    [[c, 0, (1-c)W/2], [0, c, (1-c)H/2 + delta]], which is (17) written for
    a warp of the image rather than of the landmarks.
    """
    return np.array([
        [c, 0.0, (1.0 - c) * width / 2.0],
        [0.0, c, (1.0 - c) * height / 2.0 + delta],
    ], dtype=np.float32)


def warp_image(image, c, delta):
    """Warp one frame by the affine map, with replicated borders.

    A zoom factor above one can push landmarks outside the frame, so at
    those settings drift combines a change of scale with truncation of the
    body.
    """
    import cv2

    height, width = image.shape[:2]
    return cv2.warpAffine(
        image,
        affine_matrix(c, delta, width, height),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


# --- (18) and (19), drift ------------------------------------------------

def drift(clean_predictions, perturbed_predictions):
    """D of (18), the mean absolute change of prediction over U_ho.

    The label does not change under (17), so an ideal prediction does not
    change either.
    """
    both = np.isfinite(clean_predictions) & np.isfinite(perturbed_predictions)
    if not both.any():
        return float("nan")
    return float(np.nanmean(np.abs(perturbed_predictions[both] - clean_predictions[both])))


def flip_rate(clean_predictions, perturbed_predictions, threshold=config.THRESHOLD):
    """flip of (19), the rate at which the prediction crosses tau."""
    both = np.isfinite(clean_predictions) & np.isfinite(perturbed_predictions)
    if not both.any():
        return float("nan")
    before = clean_predictions[both] <= threshold
    after = perturbed_predictions[both] <= threshold
    return float(np.mean(before != after))


def drift_interval(clean_predictions, perturbed_predictions, frames, students,
                   rows=None):
    """A bootstrap interval for D of (18), resampling students.

    Section III-C9 lists (18) among the quantities whose intervals come from
    the student bootstrap, so the per-frame changes are reduced over
    whichever students were drawn.
    """
    from . import stats

    student_of = np.array([f.student for f in frames])
    if rows is None:
        rows = np.arange(len(frames))
    rows = np.asarray(rows)

    change = np.abs(perturbed_predictions - clean_predictions)
    # A frame whose landmarks vanished under the warp has no prediction in
    # any seed, so its column is all NaN and the mean of it is undefined.
    # Those frames are dropped below rather than warned about one by one.
    with np.errstate(invalid="ignore"):
        present = np.isfinite(change).any(axis=0)
        per_frame = np.full(change.shape[1], np.nan)
        per_frame[present] = np.nanmean(change[:, present], axis=0)

    rows_of = {}
    for student in students:
        selected = rows[student_of[rows] == student]
        finite = selected[np.isfinite(per_frame[selected])]
        if finite.size:
            rows_of[student] = finite

    def statistic(resample):
        picked = [rows_of[s] for s in resample if s in rows_of]
        if not picked:
            return float("nan")
        return float(np.nanmean(per_frame[np.concatenate(picked)]))

    interval = stats.bootstrap_students(students, statistic)
    return {k: v for k, v in interval.items() if k != "draws"}


def keypoint_level_drift(frames, students, session_order, regressor_kind):
    """Drift over the grid J with (17) applied to the held-out keypoints.

    By Remark 3 this returns zero for TARE-S and TARE-FULL at every setting.
    That zero is an algebraic identity and not an empirical finding, so it is
    reported as a check on the implementation.
    """
    clean_features, _ = features.build_all(frames, session_order)
    results = {}

    for c, delta in config.perturbation_grid():
        perturbed_frames = _map_frames(frames, c, delta)
        perturbed_features, _ = features.build_all(perturbed_frames, session_order)

        for variant in config.VARIANTS:
            clean, perturbed = models.apply_to_perturbed(
                clean_features[variant], perturbed_features[variant],
                frames, students, regressor_kind,
            )
            results[(variant, c, delta)] = {
                "drift": drift(clean, perturbed),
                "flip_rate": flip_rate(clean, perturbed),
                "drift_interval": drift_interval(clean, perturbed, frames, students),
            }
    return results


def _map_frames(frames, c, delta):
    """Copies of the frames with (17) applied to their landmarks."""
    import copy

    out = []
    for frame in frames:
        clone = copy.copy(frame)
        clone.landmarks = apply_keypoint_map(frame.landmarks, c, delta)
        out.append(clone)
    return out


# --- III-C7, the image-level version -------------------------------------

def sample_held_out_frames(frames, students, seed, n_frames=config.N_WARPED_FRAMES):
    """n_img frames drawn at random from the usable frames of held-out students.

    One fold's held-out students per seed, so the warped frames and the
    predictor that scores them come from the same split.
    """
    assignment, n_folds = models.student_folds(students, seed)
    chosen = {}
    rng = np.random.default_rng(seed)
    for fold in range(n_folds):
        held = {s for s, f in assignment.items() if f == fold}
        candidates = [
            i for i, frame in enumerate(frames)
            if frame.student in held and frame.landmarks.usable
        ]
        if not candidates:
            continue
        size = min(n_frames, len(candidates))
        chosen[fold] = sorted(rng.choice(candidates, size=size, replace=False).tolist())
    return chosen


def image_level_drift(frames, students, session_order, regressor_kind,
                      batch_extract):
    """Drift with the image warped and the landmark extractor rerun.

    This version also contains detector noise, interpolation effects and
    foreshortening, which is why III-C7 calls it the informative one.
    `read_image` takes a path and returns an RGB array; `extractor` is a
    `landmarks.LandmarkExtractor`.
    """
    clean_features, _ = features.build_all(frames, session_order)
    results = {}

    for c, delta in config.perturbation_grid():
        reextracted = [frame.landmarks for frame in frames]
        selected = set()
        for seed in config.SEEDS:
            for rows in sample_held_out_frames(frames, students, seed).values():
                selected.update(rows)

        jobs = [(idx, frames[idx].image_path, c, delta)
                for idx in sorted(selected)]
        for idx, landmarks in batch_extract(jobs).items():
            reextracted[idx] = landmarks

        perturbed_frames = _replace_landmarks(frames, reextracted)
        perturbed_features, _ = features.build_all(perturbed_frames, session_order)

        for variant in config.VARIANTS:
            clean, perturbed = models.apply_to_perturbed(
                clean_features[variant], perturbed_features[variant],
                frames, students, regressor_kind,
            )
            rows = sorted(selected)
            results[(variant, c, delta)] = {
                "drift": drift(clean[:, rows], perturbed[:, rows]),
                "flip_rate": flip_rate(clean[:, rows], perturbed[:, rows]),
                "drift_interval": drift_interval(
                    clean, perturbed, frames, students, rows=rows
                ),
                "n_frames": len(rows),
            }
    return results


def _replace_landmarks(frames, landmark_list):
    import copy

    out = []
    for frame, lm in zip(frames, landmark_list):
        clone = copy.copy(frame)
        clone.landmarks = lm
        out.append(clone)
    return out


# --- III-C8, independent per-student scaling -----------------------------

def draw_zoom_factors(students, seed):
    """c_s of III-C8, drawn independently per student from F.

    F is log-uniform on [0.64, 1.5625], the zoom range of J without its two
    outermost levels.
    """
    low, high = config.ZOOM_RANGE
    rng = np.random.default_rng(seed)
    draws = np.exp(rng.uniform(np.log(low), np.log(high), size=len(students)))
    return dict(zip(sorted(students), draws.tolist()))


def warp_cohort(frames, factor_of_student, batch_extract):
    """Warp every frame of each student by that student's factor and re-extract.

    Used by both parts of III-C8: with one common factor for the
    dose-response curve of (20), and with independent factors for (21).

    `batch_extract` takes (index, path, c, delta) jobs and returns a map from
    index to z_i, so the whole cohort goes out in one call and can be spread
    over several processes.
    """
    jobs = []
    for index, frame in enumerate(frames):
        c = factor_of_student.get(frame.student)
        if c is not None and frame.landmarks.usable:
            jobs.append((index, frame.image_path, c, 0.0))

    extracted = batch_extract(jobs)
    landmark_list = [
        extracted.get(index, frame.landmarks)
        for index, frame in enumerate(frames)
    ]
    return _replace_landmarks(frames, landmark_list)


def dose_response(frames, students, session_order, regressor_kind,
                  batch_extract, on_level=None):
    """Gamma of (20), with one common factor c applied to all students.

    Gamma(c) is the mean over students of e_s(c) minus e_s(1), so it is
    paired by student. It is summarised as gamma_0 + gamma_1 ln c, and the
    slope gamma_1 is what the fit reports.
    """
    from . import audit

    clean_features, _ = features.build_all(frames, session_order)
    baseline_error = {}
    for variant in config.VARIANTS:
        predictions = models.out_of_fold_predictions(
            clean_features[variant], frames, students, regressor_kind
        )
        error = audit.frame_errors(predictions, frames)
        baseline_error[variant] = audit.student_errors(error, frames, students)

    curves = {variant: {} for variant in config.VARIANTS}
    per_student = {variant: {} for variant in config.VARIANTS}

    for c in config.zoom_levels():
        common = {s: c for s in students}
        warped_frames = warp_cohort(frames, common, batch_extract)
        warped_features, _ = features.build_all(warped_frames, session_order)

        for variant in config.VARIANTS:
            predictions = models.out_of_fold_predictions(
                warped_features[variant], warped_frames, students, regressor_kind
            )
            error = audit.frame_errors(predictions, warped_frames)
            student_error = audit.student_errors(error, warped_frames, students)
            gains = {
                s: student_error[s] - baseline_error[variant][s] for s in students
            }
            per_student[variant][c] = gains
            curves[variant][c] = float(np.nanmean(list(gains.values())))
        if on_level:
            on_level(c, curves, per_student, baseline_error)

    fits = {}
    for variant in config.VARIANTS:
        fit = log_linear_fit(curves[variant])
        fit["slope_interval"] = bootstrap_slope(per_student[variant], students)
        fits[variant] = fit
    return {"curve": curves, "per_student": per_student, "fit": fits,
            "clean_student_error": baseline_error}


def bootstrap_slope(per_student_gains, students):
    """A student-level bootstrap interval for gamma_1.

    Section III-C8 asks for the slope of the log-linear summary to be
    reported with such an interval, and III-C9 makes the student the
    independent unit, so the resample draws students and the curve is
    rebuilt from the gains of whoever was drawn.
    """
    from . import stats

    def slope(resample):
        curve = {}
        for c, gains in per_student_gains.items():
            values = [gains[s] for s in resample if s in gains]
            if values:
                curve[c] = float(np.nanmean(values))
        return log_linear_fit(curve)["slope"]

    return stats.bootstrap_students(students, slope)


def log_linear_fit(curve):
    """gamma_0 and gamma_1 of the summary Gamma(c) = gamma_0 + gamma_1 ln c."""
    factors = sorted(curve)
    x = np.log(factors)
    y = np.array([curve[c] for c in factors])
    finite = np.isfinite(y)
    if finite.sum() < 2:
        return {"intercept": float("nan"), "slope": float("nan")}
    slope, intercept = np.polyfit(x[finite], y[finite], 1)
    return {"intercept": float(intercept), "slope": float(slope)}


def perturbed_cohort(frames, students, session_order, regressor_kind,
                     batch_extract, proxies,
                     n_replicates=config.N_REPLICATES,
                     start_at=0, on_replicate=None):
    """Equation (21) and the matching gaps, over n_rep replicates.

    Each replicate draws an independent factor per student, warps every
    frame of that student, re-extracts, and scores the held-out students
    with predictors trained on the clean data of the training students. The
    perturbed apparent scale is c_s P_s, and the groups are rebuilt from it
    by the construction used for G(s).
    """
    from . import audit

    clean_features, _ = features.build_all(frames, session_order)
    replicates = []

    for b in range(start_at, n_replicates):
        factors = draw_zoom_factors(students, seed=1000 + b)
        warped_frames = warp_cohort(frames, factors, batch_extract)
        warped_features, _ = features.build_all(warped_frames, session_order)

        entry = {"replicate": b, "factors": factors, "variants": {}}
        for variant in config.VARIANTS:
            clean, perturbed = models.apply_to_perturbed(
                clean_features[variant], warped_features[variant],
                frames, students, regressor_kind,
            )
            error = audit.frame_errors(perturbed, warped_frames)
            student_error = audit.student_errors(error, warped_frames, students)

            record = {
                "student_error": student_error,
                "e_spread": float(
                    np.nanmax(list(student_error.values()))
                    - np.nanmin(list(student_error.values()))
                ),
            }
            for proxy_name, proxy in proxies.items():
                shifted = {s: factors[s] * proxy[s] for s in proxy if s in factors}
                groups = audit.scale_groups(shifted)
                record[proxy_name] = {
                    "rho": audit.rank_association(student_error, shifted),
                    "eo_reg": audit.equalized_odds_regression(
                        error, warped_frames, groups
                    )["gap"],
                    "eo_bin": audit.equalized_odds_binary(
                        perturbed, warped_frames, groups, config.THRESHOLD
                    )["gap"],
                }
            entry["variants"][variant] = record
        replicates.append(entry)
        if on_replicate:
            on_replicate(b, entry)
    return {"replicates": replicates,
            "distribution": replicate_distribution(replicates, proxies)}


def replicate_distribution(replicates, proxies):
    """The distribution of (21) over b, which is what III-C8 reports.

    Reported next to the clean values of rho and e_spread, for every
    variant, so the summary keeps the same keys the clean audit uses.
    """
    summary = {}
    for variant in config.VARIANTS:
        record = {"e_spread": _spread_over_replicates(
            [r["variants"][variant]["e_spread"] for r in replicates]
        )}
        for proxy_name in proxies:
            record[proxy_name] = {
                key: _spread_over_replicates(
                    [r["variants"][variant][proxy_name][key] for r in replicates]
                )
                for key in ("rho", "eo_reg", "eo_bin")
            }
        summary[variant] = record
    return summary


def _spread_over_replicates(values):
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if finite.size == 0:
        return {"mean": float("nan"), "q05": float("nan"), "q95": float("nan"),
                "n_replicates": 0}
    return {
        "mean": float(finite.mean()),
        "q05": float(np.quantile(finite, 0.05)),
        "q95": float(np.quantile(finite, 0.95)),
        "n_replicates": int(finite.size),
    }
