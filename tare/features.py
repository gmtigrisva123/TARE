"""The three feature sets of III-C4, built from one extraction.

RAW is the control of III-C4a and keeps every length in pixels, with the
keypoint coordinates divided by the image width and height. Dividing by
image size removes the sensor resolution and nothing else, so body size and
seating distance stay in every length.

TARE-S is stage one of III-C4c. Coordinates become offsets from the
shoulder midpoint in units of the windowed in-frame scale, and lengths
become ratios to it. The ratio w_i / L_i is left out because w_i already
forms part of L_i, which is why RAW has 33 entries and TARE-S has 32. The
six angles are carried over unchanged.

TARE-FULL is stage H2 of III-C4d, TARE-S minus the student's own baseline,
the componentwise median over the first 60 s of their first session. No
label enters that median, so the calibration frames stay in the scored set.
"""

import numpy as np

from . import config, geometry


def raw_feature_names():
    """The 33 entries of phi^RAW, in the order III-C4a lists them."""
    names = []
    for name in geometry.KEYPOINT_NAMES:
        names += [f"{name}_x_over_W", f"{name}_y_over_H"]
    names += ["iod_px", "shoulder_width_px", "nose_to_mid_px", "face_height_px"]
    names += ["face_center_height_px"]
    names += ["head_pitch", "head_yaw", "head_roll",
              "shoulder_tilt", "neck_tilt", "ear_asymmetry"]
    return names


def tare_s_feature_names():
    """The 32 entries of phi^TARE-S, in the order III-C4c lists them."""
    names = []
    for name in geometry.KEYPOINT_NAMES:
        names += [f"{name}_x_rel", f"{name}_y_rel"]
    names += ["iod_over_L", "face_height_over_L",
              "nose_to_mid_over_L", "face_center_height_over_L"]
    names += ["head_pitch", "head_yaw", "head_roll",
              "shoulder_tilt", "neck_tilt", "ear_asymmetry"]
    return names


N_RAW = len(raw_feature_names())            # 33
N_TARE = len(tare_s_feature_names())        # 32


def _angles(landmarks):
    kp = landmarks.keypoints
    return [
        landmarks.head_angles[0],
        landmarks.head_angles[1],
        landmarks.head_angles[2],
        geometry.shoulder_tilt(kp),
        geometry.neck_tilt(kp),
        geometry.ear_asymmetry(kp),
    ]


def raw_vector(landmarks, width=config.IMAGE_WIDTH, height=config.IMAGE_HEIGHT):
    """phi^RAW for one frame, or NaN throughout when u_i = 0."""
    if not landmarks.usable:
        return np.full(N_RAW, np.nan)

    kp = landmarks.keypoints
    values = []
    for point in kp:
        values += [point[0] / width, point[1] / height]
    values += [
        landmarks.iod,
        landmarks.shoulder_width,
        geometry.nose_to_shoulder_midpoint(kp),
        landmarks.face_height,
        geometry.face_center_height(kp),
    ]
    values += _angles(landmarks)
    return np.asarray(values, dtype=float)


def tare_s_vector(landmarks, windowed_scale):
    """phi^TARE-S for one frame, by (4) and (5).

    `windowed_scale` is L-bar of (4), the median of L over the window of
    five frames of the same session.
    """
    if not landmarks.usable or not np.isfinite(windowed_scale) or windowed_scale <= 0:
        return np.full(N_TARE, np.nan)

    kp = landmarks.keypoints
    mid = geometry.shoulder_midpoint(kp)
    values = []
    for point in kp:
        values += [
            (point[0] - mid[0]) / windowed_scale,
            (point[1] - mid[1]) / windowed_scale,
        ]
    values += [
        landmarks.iod / windowed_scale,
        landmarks.face_height / windowed_scale,
        geometry.nose_to_shoulder_midpoint(kp) / windowed_scale,
        geometry.face_center_height(kp) / windowed_scale,
    ]
    values += _angles(landmarks)
    return np.asarray(values, dtype=float)


def in_frame_scales(frames):
    """L_i and L-bar_i for every frame, by (4).

    The median window holds five frames of the same session. Alignment is a
    setting because III-C4c leaves it open: Table II describes the window as
    sitting around frame i, which is the centered reading and the default
    here, while a streaming deployment would use the trailing one.
    """
    scale = np.full(len(frames), np.nan)
    branch = [None] * len(frames)
    for idx, frame in enumerate(frames):
        lm = frame.landmarks
        if lm.usable:
            scale[idx] = geometry.in_frame_scale(
                lm.shoulder_width, lm.iod, config.EYE_TERM_WEIGHT
            )
            branch[idx] = geometry.active_scale_branch(
                lm.shoulder_width, lm.iod, config.EYE_TERM_WEIGHT
            )

    windowed = np.full(len(frames), np.nan)
    half = config.MEDIAN_WINDOW // 2
    for positions in _session_positions(frames).values():
        for slot, idx in enumerate(positions):
            if config.MEDIAN_WINDOW_ALIGNMENT == "centered":
                lo, hi = slot - half, slot + half + 1
            elif config.MEDIAN_WINDOW_ALIGNMENT == "trailing":
                lo, hi = slot - config.MEDIAN_WINDOW + 1, slot + 1
            else:
                raise ValueError(
                    f"Unknown window alignment {config.MEDIAN_WINDOW_ALIGNMENT!r}."
                )
            lo = max(lo, 0)
            neighbours = [scale[positions[j]] for j in range(lo, min(hi, len(positions)))]
            present = [v for v in neighbours if np.isfinite(v)]
            if present:
                windowed[idx] = float(np.median(present))
    return scale, windowed, branch


def _session_positions(frames):
    """Frame indices grouped by session and ordered in time."""
    sessions = {}
    for idx, frame in enumerate(frames):
        sessions.setdefault(frame.session, []).append(idx)
    for key in sessions:
        sessions[key].sort(key=lambda i: frames[i].time_s)
    return sessions


def scale_branch_shares(branch):
    """The share of usable frames in which each branch of (4) is active.

    Section III-C4c asks for this alongside the validation of lambda = 4.5.
    """
    present = [b for b in branch if b is not None]
    if not present:
        return {}
    total = len(present)
    return {
        "shoulder": present.count("shoulder") / total,
        "eye": present.count("eye") / total,
        "n_usable_frames": total,
    }


def student_baselines(frames, tare_s, session_order):
    """beta_s of (6), the componentwise median over the calibration frames.

    C_s holds the frames in the first 60 s of the earliest session of each
    student. One baseline serves every session of that student, so no
    activity information leaks into it, and no label is read.

    A student with no usable calibration frame has no baseline, and
    III-C4d says they are represented by phi^TARE-S, so their baseline is
    returned as zero and flagged.
    """
    baselines, calibrated = {}, {}
    for student, activities in session_order.items():
        first = activities[0]
        rows = [
            tare_s[idx] for idx, frame in enumerate(frames)
            if frame.student == student
            and frame.activity == first
            and frame.time_s < config.CALIBRATION_SECONDS
            and np.isfinite(tare_s[idx]).any()
        ]
        if rows:
            baselines[student] = np.nanmedian(np.stack(rows), axis=0)
            calibrated[student] = True
        else:
            baselines[student] = np.zeros(N_TARE)
            calibrated[student] = False
    return baselines, calibrated


def tare_full_matrix(frames, tare_s, baselines):
    """phi^TARE-FULL for every frame, by (6)."""
    out = np.empty_like(tare_s)
    for idx, frame in enumerate(frames):
        out[idx] = tare_s[idx] - baselines[frame.student]
    return out


def build_all(frames, session_order):
    """The three feature matrices, plus what (4) and (6) reported on the way.

    Called once per extraction. The same landmarks feed all three variants,
    which is what makes the comparison of III-A controlled.
    """
    scale, windowed, branch = in_frame_scales(frames)

    raw = np.stack([raw_vector(f.landmarks) for f in frames])
    tare_s = np.stack([
        tare_s_vector(f.landmarks, windowed[i]) for i, f in enumerate(frames)
    ])
    baselines, calibrated = student_baselines(frames, tare_s, session_order)
    tare_full = tare_full_matrix(frames, tare_s, baselines)

    return {
        "RAW": raw,
        "TARE-S": tare_s,
        "TARE-FULL": tare_full,
    }, {
        "in_frame_scale": scale,
        "windowed_scale": windowed,
        "scale_branch": branch,
        "branch_shares": scale_branch_shares(branch),
        "baselines": baselines,
        "has_baseline": calibrated,
    }
