"""MediaPipe extraction, producing the raw landmark quantities z_i.

Section III-B1 fixes two things. Landmarks come from a fresh run of the
MediaPipe pose and face landmarkers, never from the annotations DIPSER
ships, because the two sources agree at r = 0.99 on interocular distance and
only r = 0.49 on shoulder width and an agreement that size does not say
which one is wrong. And nothing is mixed between sources, so one extractor
supplies every quantity.

z_i is the tuple III-C2 names: the eleven keypoints, w_i, IOD_i, h_i and the
three head angles, plus the indicator u_i, which is one when both a face and
a body are detected.
"""

import os
from dataclasses import dataclass

import numpy as np

from . import geometry


@dataclass
class RawLandmarks:
    """z_i for one frame, with the detection indicator.

    `keypoints` is 11 by 2 in pixels. When `usable` is False the fields hold
    NaN and the frame still gets scored, by III-C5, with the training mean
    as its prediction.
    """

    keypoints: np.ndarray
    shoulder_width: float
    iod: float
    face_height: float
    head_angles: np.ndarray      # pitch, yaw, roll in degrees
    usable: bool

    @staticmethod
    def missing():
        return RawLandmarks(
            keypoints=np.full((geometry.N_KEYPOINTS, 2), np.nan),
            shoulder_width=np.nan,
            iod=np.nan,
            face_height=np.nan,
            head_angles=np.full(3, np.nan),
            usable=False,
        )


# Indices of the eleven keypoints inside the 33 BlazePose pose landmarks,
# in the order geometry.KEYPOINT_NAMES uses.
_BLAZEPOSE_INDEX = {
    "nose": 0,
    "left_eye": 2, "right_eye": 5,
    "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
}

# Iris centers in the 478-landmark face mesh. The last ten landmarks are the
# two irises, five points each, and the first of each five is the center.
_LEFT_IRIS_CENTER = 468
_RIGHT_IRIS_CENTER = 473

# Face height is read between the forehead and the chin of the face mesh.
_FOREHEAD = 10
_CHIN = 152


class LandmarkExtractor:
    """One MediaPipe pose landmarker and one face landmarker.

    Both task files have to be downloaded once from the MediaPipe model
    index and their paths given here. The extractor is created once and
    reused, because III-G runs it once per sampled frame and feeds the same
    extraction to all three variants.
    """

    def __init__(self, pose_task_path, face_task_path):
        for path in (pose_task_path, face_task_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"MediaPipe task file not found: {path}. Download "
                    "pose_landmarker.task and face_landmarker.task from the "
                    "MediaPipe model index and pass their paths."
                )
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        self._vision = mp_vision

        pose_options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=pose_task_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,                 # one student per frame, assumption A3
        )
        face_options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=face_task_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            output_facial_transformation_matrixes=True,   # head angles come from this
        )
        self._pose = mp_vision.PoseLandmarker.create_from_options(pose_options)
        self._face = mp_vision.FaceLandmarker.create_from_options(face_options)

    def __call__(self, image_rgb):
        """Run both landmarkers on one RGB frame and return z_i."""
        height, width = image_rgb.shape[:2]
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(image_rgb),
        )

        pose_result = self._pose.detect(mp_image)
        face_result = self._face.detect(mp_image)

        body_found = bool(pose_result.pose_landmarks)
        face_found = bool(face_result.face_landmarks)
        if not (body_found and face_found):
            return RawLandmarks.missing()

        keypoints = self._read_keypoints(pose_result.pose_landmarks[0], width, height)
        face = face_result.face_landmarks[0]

        iod = self._pixel_distance(face, _LEFT_IRIS_CENTER, _RIGHT_IRIS_CENTER, width, height)
        face_height = self._pixel_distance(face, _FOREHEAD, _CHIN, width, height)
        head_angles = self._head_angles(face_result)

        return RawLandmarks(
            keypoints=keypoints,
            shoulder_width=geometry.shoulder_width(keypoints),
            iod=iod,
            face_height=face_height,
            head_angles=head_angles,
            usable=True,
        )

    @staticmethod
    def _read_keypoints(pose_landmarks, width, height):
        keypoints = np.empty((geometry.N_KEYPOINTS, 2), dtype=float)
        for slot, name in enumerate(geometry.KEYPOINT_NAMES):
            landmark = pose_landmarks[_BLAZEPOSE_INDEX[name]]
            keypoints[slot] = (landmark.x * width, landmark.y * height)
        return keypoints

    @staticmethod
    def _pixel_distance(face_landmarks, index_a, index_b, width, height):
        a = face_landmarks[index_a]
        b = face_landmarks[index_b]
        dx = (a.x - b.x) * width
        dy = (a.y - b.y) * height
        return float(np.hypot(dx, dy))

    @staticmethod
    def _head_angles(face_result):
        """Pitch, yaw and roll in degrees, from the face transformation matrix.

        The matrix is the 4 by 4 pose of the head. Its upper left 3 by 3
        block is the rotation, and the three angles are read off it as an
        x-y-z Tait-Bryan sequence.
        """
        matrices = getattr(face_result, "facial_transformation_matrixes", None)
        if not matrices:
            return np.full(3, np.nan)
        rotation = np.asarray(matrices[0])[:3, :3]

        sy = float(np.hypot(rotation[0, 0], rotation[1, 0]))
        if sy > 1e-6:
            pitch = np.arctan2(rotation[2, 1], rotation[2, 2])
            yaw = np.arctan2(-rotation[2, 0], sy)
            roll = np.arctan2(rotation[1, 0], rotation[0, 0])
        else:
            pitch = np.arctan2(-rotation[1, 2], rotation[1, 1])
            yaw = np.arctan2(-rotation[2, 0], sy)
            roll = 0.0
        return np.degrees([pitch, yaw, roll]).astype(float)

    def close(self):
        self._pose.close()
        self._face.close()


def detection_rate_by_activity(frames):
    """The share of usable frames per activity.

    Section III-B1 reports 98.7% on lecture frames against 78.4% on robotics
    frames and calls it a property of the landmarker on these data rather
    than an effect of TARE, so the figure is checked rather than assumed.
    """
    totals, usable = {}, {}
    for frame in frames:
        totals[frame.activity] = totals.get(frame.activity, 0) + 1
        usable[frame.activity] = usable.get(frame.activity, 0) + int(frame.landmarks.usable)
    return {a: usable[a] / totals[a] for a in sorted(totals)}
