"""Geometric references built on the eleven upper-body keypoints.

Section III-C2 gives the shoulder midpoint and the shoulder width in (2) and
then says the face center, the shoulder tilt, the neck tilt and the ear
asymmetry "are computed from the same landmarks" without writing the
formulas. The paper marks that as pending. The definitions below follow the
anatomical descriptions supplied by the author, written against the eleven
keypoints III-C2 lists, and each one records which description it comes
from.

Two constraints from the paper decide how the three angles are written.
Proposition 1 holds only if an angle is unchanged by a positive scaling and
a translation, and the RAW set of III-C4a counts all three of them among its
six angles rather than among its pixel lengths. Each angle below is
therefore a ratio of keypoint differences, which is invariant to both.

Remark 3 calls for a second property. Keypoint-level drift under (17) has to
be zero for TARE-S by algebra, so every quantity the features use has to
move with the keypoints. The face center is read off the five face keypoints
for that reason, not off the 478 face landmarks.
"""

import numpy as np

# Order of the eleven upper-body keypoints of III-C2, in the BlazePose
# convention: the nose, then the left and right eyes, ears, shoulders,
# elbows and wrists. Hips and legs are left out because the desk occludes
# them in this camera configuration.
KEYPOINT_NAMES = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
)
N_KEYPOINTS = len(KEYPOINT_NAMES)
KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# The five keypoints that lie on the face. The nose sits on the midline and
# the eyes and ears come in left and right pairs, so averaging all five
# lands on the midline again.
FACE_KEYPOINTS = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")


def shoulder_midpoint(keypoints):
    """m_i of (2), the midpoint of the two shoulders, in pixels."""
    left = keypoints[KP["left_shoulder"]]
    right = keypoints[KP["right_shoulder"]]
    return 0.5 * (left + right)


def shoulder_width(keypoints):
    """w_i of (2), the distance between the two shoulders, in pixels."""
    left = keypoints[KP["left_shoulder"]]
    right = keypoints[KP["right_shoulder"]]
    return float(np.linalg.norm(left - right))


def face_center(keypoints):
    """A point on the vertical midline of the face, in pixels.

    The author's definition is the axis that splits the face into two
    symmetrical halves and runs down through the glabella, the bridge and
    tip of the nose, the philtrum and the center of the chin. None of those
    anatomical points is a keypoint on its own. What is available is the
    nose, which sits on the axis, and the eye and ear pairs, which sit
    symmetrically on either side of it. Averaging the five therefore returns
    a point on the axis, with the left and right members of each pair
    cancelling.

    The RAW set uses this point once, for the height of the face center
    above the shoulder midpoint.
    """
    pts = np.stack([keypoints[KP[name]] for name in FACE_KEYPOINTS])
    return pts.mean(axis=0)


def face_center_height(keypoints):
    """How far the face center sits above m_i, in pixels.

    Image rows grow downward, so the shoulder row minus the face row is
    positive while the head is above the shoulders.
    """
    return float(shoulder_midpoint(keypoints)[1] - face_center(keypoints)[1])


def nose_to_shoulder_midpoint(keypoints):
    """The distance from the nose to m_i, in pixels. Used by III-C4a."""
    return float(np.linalg.norm(keypoints[KP["nose"]] - shoulder_midpoint(keypoints)))


def shoulder_tilt(keypoints):
    """The slope of the shoulder line against the horizontal, in degrees.

    From the author's definition: the line joining the outer tips of the two
    shoulders is level in an upright resting posture, and a tilt is one
    shoulder being carried higher than the other. The angle is signed, and
    the sign says which shoulder is the high one. Measured from the left
    shoulder to the right one so the sign stays consistent across frames.

    Image rows grow downward, so the row difference is negated to make a
    raised right shoulder give a positive angle.
    """
    left = keypoints[KP["left_shoulder"]]
    right = keypoints[KP["right_shoulder"]]
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    return float(np.degrees(np.arctan2(-dy, dx)))


def neck_tilt(keypoints):
    """How far the neck axis leans away from vertical, in degrees.

    From the author's definition: the cervical spine deviating sideways from
    vertical, seen from the front as the head leaning toward one shoulder.
    The definition also notes that computer vision often reads this off head
    roll, but roll is already one of the three head angles of III-C2 and the
    paper counts neck tilt as a sixth angle beside it, so this is measured
    from the keypoints instead.

    The axis runs from the shoulder midpoint up to the face center, both of
    which sit on the body midline. The angle returned is that axis against
    the image vertical, signed, so a lean toward one shoulder and a lean
    toward the other have opposite signs.
    """
    axis = face_center(keypoints) - shoulder_midpoint(keypoints)
    dx = axis[0]
    dy = -axis[1]          # upward is positive once rows are flipped
    return float(np.degrees(np.arctan2(dx, dy)))


def ear_asymmetry(keypoints):
    """The slope of the line joining the two ears, in degrees.

    The author's definition lists height asymmetry first among the ways ear
    asymmetry shows up: one ear sitting higher on the skull than the other.
    Projection and morphology are the other two, and neither is recoverable
    from a pair of 2D ear keypoints. Height asymmetry is, and written as the
    angle of the ear line against the horizontal it stays invariant to
    scaling and translation, which is what the six angles of III-C4a have to
    be.

    Measured from the left ear to the right one, with rows flipped, on the
    same convention as `shoulder_tilt`.
    """
    left = keypoints[KP["left_ear"]]
    right = keypoints[KP["right_ear"]]
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    return float(np.degrees(np.arctan2(-dy, dx)))


def in_frame_scale(shoulder_w, iod, eye_weight):
    """L_i of (4), the larger of the shoulder term and the weighted eye term.

    Both arguments exist on every usable frame, which III-C4c relies on when
    it says the maximum never receives a missing argument, so this is a
    plain maximum. The shoulder term wins when w_i >= lambda * IOD_i and the
    eye term wins otherwise.
    """
    return float(max(shoulder_w, eye_weight * iod))


def active_scale_branch(shoulder_w, iod, eye_weight):
    """Which term of (4) set L_i. Reported for the share-of-frames check
    that III-C4c asks for.
    """
    return "shoulder" if shoulder_w >= eye_weight * iod else "eye"
