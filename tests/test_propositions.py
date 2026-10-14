"""Checks on the algebraic claims of Section III.

Proposition 1, Proposition 2 and Remark 3 are identities, so they hold for
any numbers and can be checked without data. Remark 3 says so itself: it
calls keypoint-level drift a unit test. Nothing here is a research result
and no file in this directory produces one. The arrays below are arbitrary
inputs chosen to exercise the algebra, in the way any unit test picks
arguments.

Run with:
    python -m pytest tests -q
"""

import numpy as np
import pytest

from tare import config, features, geometry, zoom
from tare.landmarks import RawLandmarks


def a_frame(seed):
    """An arbitrary usable z_i. The numbers carry no meaning."""
    rng = np.random.default_rng(seed)
    keypoints = rng.uniform(50, 400, size=(geometry.N_KEYPOINTS, 2))
    return RawLandmarks(
        keypoints=keypoints,
        shoulder_width=geometry.shoulder_width(keypoints),
        iod=float(rng.uniform(20, 40)),
        face_height=float(rng.uniform(80, 160)),
        head_angles=rng.uniform(-30, 30, size=3),
        usable=True,
    )


class FakeFrame:
    """The two fields `features.build_all` reads off a frame."""

    def __init__(self, student, activity, time_s, landmarks):
        self.student = student
        self.activity = activity
        self.time_s = time_s
        self.landmarks = landmarks

    @property
    def session(self):
        return (self.student, self.activity)


def a_session(student, n_frames, seed, activity=config.LECTURE_ACTIVITY):
    return [
        FakeFrame(student, activity, i * config.SAMPLING_BIN_SECONDS, a_frame(seed + i))
        for i in range(n_frames)
    ]


# --- Feature dimensions of III-C4 ----------------------------------------

def test_feature_dimensions_match_the_paper():
    assert features.N_RAW == 33
    assert features.N_TARE == 32


def test_shoulder_ratio_is_not_a_tare_feature():
    """w_i / L-bar_i is left out because w_i already forms part of L_i."""
    names = features.tare_s_feature_names()
    assert not any(name.startswith("shoulder_width") for name in names)


# --- Proposition 1 -------------------------------------------------------

@pytest.mark.parametrize("c", [0.5, 0.8, 1.3, 2.0])
@pytest.mark.parametrize("shift", [0.0, -40.0, 25.0])
def test_proposition_1_exact_invariance(c, shift):
    """phi^TARE-S is unchanged when one similarity map acts on the window.

    Every keypoint and every landmark length is transformed by T(p) = cp + eta
    with c > 0 and the head angles are left alone, which is the hypothesis of
    Proposition 1 and of assumption A2.
    """
    order = {"s1": [config.LECTURE_ACTIVITY]}
    frames = a_session("s1", 9, seed=1)
    sets_before, _ = features.build_all(frames, order)

    warped = [
        FakeFrame(f.student, f.activity, f.time_s,
                  zoom.apply_keypoint_map(f.landmarks, c, shift))
        for f in frames
    ]
    sets_after, _ = features.build_all(warped, order)

    np.testing.assert_allclose(
        sets_before["TARE-S"], sets_after["TARE-S"], rtol=1e-9, atol=1e-9
    )


def test_raw_features_do_move_under_the_same_map():
    """The control is not invariant, which is why it is the control."""
    frames = a_session("s1", 9, seed=2)
    sets_before, _ = features.build_all(frames, {"s1": [config.LECTURE_ACTIVITY]})
    warped = [
        FakeFrame(f.student, f.activity, f.time_s,
                  zoom.apply_keypoint_map(f.landmarks, 1.5, 0.0))
        for f in frames
    ]
    sets_after, _ = features.build_all(warped, {"s1": [config.LECTURE_ACTIVITY]})
    assert not np.allclose(sets_before["RAW"], sets_after["RAW"])


def test_angles_are_invariant_to_scaling_and_translation():
    """The last line of the proof of Proposition 1.

    Covers the three angles whose definitions Section III-B2 left pending,
    because the RAW set counts them among its angles and the proof needs
    them to survive the map.
    """
    keypoints = a_frame(3).keypoints
    moved = 2.4 * keypoints + np.array([17.0, -9.0])
    for angle in (geometry.shoulder_tilt, geometry.neck_tilt, geometry.ear_asymmetry):
        assert angle(keypoints) == pytest.approx(angle(moved), abs=1e-9)


# --- Proposition 2 -------------------------------------------------------

def test_proposition_2_calibration_frames_are_centered():
    """The componentwise median of phi^TARE-FULL over C_s is zero."""
    n = int(config.CALIBRATION_SECONDS / config.SAMPLING_BIN_SECONDS) + 6
    frames = a_session("s1", n, seed=4)
    order = {"s1": [config.LECTURE_ACTIVITY]}

    sets, diagnostics = features.build_all(frames, order)
    rows = [
        i for i, f in enumerate(frames)
        if f.time_s < config.CALIBRATION_SECONDS
        and np.isfinite(sets["TARE-S"][i]).any()
    ]
    assert rows, "the calibration window has to hold some usable frame"
    assert diagnostics["has_baseline"]["s1"]

    medians = np.nanmedian(sets["TARE-FULL"][rows], axis=0)
    np.testing.assert_allclose(medians, np.zeros(features.N_TARE), atol=1e-9)


def test_student_without_calibration_falls_back_to_tare_s():
    """III-C4d: a student with no baseline is represented by phi^TARE-S."""
    frames = a_session("s1", 4, seed=5)
    for frame in frames:
        frame.time_s += config.CALIBRATION_SECONDS * 2      # nothing inside C_s
    sets, diagnostics = features.build_all(frames, {"s1": [config.LECTURE_ACTIVITY]})
    assert diagnostics["has_baseline"]["s1"] is False
    np.testing.assert_allclose(sets["TARE-FULL"], sets["TARE-S"], atol=1e-12)


# --- Remark 3 ------------------------------------------------------------

@pytest.mark.parametrize("pair", config.perturbation_grid())
def test_remark_3_keypoint_drift_vanishes_for_both_tare_variants(pair):
    """Zero drift under (17) is an algebraic identity, not a finding.

    Checked on the features rather than on a prediction, because a predictor
    is a function of the features: identical features give an identical
    prediction, so zero feature change is the stronger statement.
    """
    c, delta = pair
    order = {"s1": [config.LECTURE_ACTIVITY]}
    frames = a_session("s1", 12, seed=6)
    sets_before, _ = features.build_all(frames, order)

    warped = [
        FakeFrame(f.student, f.activity, f.time_s,
                  zoom.apply_keypoint_map(f.landmarks, c, delta))
        for f in frames
    ]
    sets_after, _ = features.build_all(warped, order)

    for variant in ("TARE-S", "TARE-FULL"):
        np.testing.assert_allclose(
            sets_before[variant], sets_after[variant], rtol=1e-9, atol=1e-9
        )


# --- Table III and the grids ---------------------------------------------

def test_perturbation_grid_matches_table_iii():
    """Six zoom levels and two pure shifts, seven settings with the clean one."""
    grid = config.perturbation_grid()
    zooms = sorted(c for c, d in grid if d == 0.0)
    shifts = sorted(d for c, d in grid if c == 1.0)
    assert len(grid) == 8
    assert len(zooms) == 6
    assert shifts == [-40.0, 40.0]
    assert zooms[0] == pytest.approx(1.25 ** -3)
    assert zooms[-1] == pytest.approx(1.25 ** 3)


def test_zoom_distribution_drops_the_two_outermost_levels():
    """F is log-uniform on the zoom range of J without its outermost levels."""
    low, high = config.ZOOM_RANGE
    assert low == pytest.approx(1.25 ** -2)
    assert high == pytest.approx(1.25 ** 2)


def test_affine_matrix_agrees_with_the_keypoint_map():
    """The image warp of III-C7 is (17) written for the image.

    Both move a point the same way, which is what makes the keypoint-level
    and image-level versions two readings of one perturbation.
    """
    c, delta = 1.25, 17.0
    point = np.array([123.0, 211.0])
    matrix = zoom.affine_matrix(c, delta)
    by_matrix = matrix @ np.array([point[0], point[1], 1.0])

    o = zoom.image_center()
    by_map = c * (point - o) + o + np.array([0.0, delta])
    np.testing.assert_allclose(by_matrix, by_map, atol=1e-6)


# --- III-C9 --------------------------------------------------------------

def test_primary_family_has_ten_comparisons():
    """TARE-S and TARE-FULL against RAW on ERROR, EO_reg and EO_bin(2.5),
    the last two for each of the two proxies."""
    from tare import stats

    assert len(stats.primary_family_names()) == 10


def test_sign_flip_floor_is_two_to_the_minus_n():
    """The smallest attainable one-sided p-value with N students."""
    from tare import stats

    gains = [0.3] * 16          # every student favours the variant
    result = stats.sign_flip_test(gains)
    assert result["exact"] is True
    assert result["p_value"] == pytest.approx(2.0 ** -16)


def test_equivalence_uses_the_margin_of_table_iii():
    from tare import stats

    inside = stats.equivalence({"low": -0.02, "high": 0.03})
    outside = stats.equivalence({"low": -0.02, "high": 0.08})
    assert inside["unchanged"] is True
    assert outside["unchanged"] is False
    assert inside["margin"] == 0.05
